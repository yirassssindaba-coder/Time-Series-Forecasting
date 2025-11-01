# Time Series Forecasting — Quick Start

Ringkasan  
Berisi panduan singkat untuk menjalankan skrip PowerShell `run_pipeline_complete_with_ui.ps1` yang akan:
- Menulis pipeline Python (`pipeline_run.py`).
- Menulis dua halaman HTML interaktif:
  - `notebooks_html/report_combined_yoy_and_notebook.html` (gabungan image + YoY editor)
  - `yoy_analysis_with_explanation.html` (YoY standalone editor)
- Membuat struktur folder hasil (data, models, notebooks_html, results).
- (Opsional) Menjalankan pipeline Python untuk mengunduh data dan menghasilkan laporan.

Lokasi default hasil
- Default proyek: `C:\Users\<Anda>\Desktop\time-series-forecasting`
- Folder hasil: `C:\Users\<Anda>\Desktop\time-series-forecasting\hasil`
Anda dapat mengganti lokasi saat menjalankan skrip.

Prerequisites
- Windows / macOS / Linux dengan PowerShell (pwsh) atau Windows PowerShell.
- Python 3.8+ tersedia di PATH.
- Rekomendasi paket (jika ingin menjalankan pipeline Python):
  - pandas, numpy, matplotlib, plotly, joblib, scikit-learn, lightgbm, yfinance, requests

Instalasi paket Python (opsional — hanya bila ingin menjalankan pipeline)
Buka terminal / PowerShell dan jalankan:
```
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib plotly joblib scikit-learn lightgbm yfinance requests
```

Menjalankan skrip PowerShell (menulis semua file)
- Simpan file `run_pipeline_complete_with_ui.ps1` pada komputer Anda.
- Jalankan:
```
pwsh -ExecutionPolicy Bypass -File .\run_pipeline_complete_with_ui.ps1
```
atau jika Anda tidak ingin pipeline Python dieksekusi otomatis:
```
pwsh -ExecutionPolicy Bypass -File .\run_pipeline_complete_with_ui.ps1 -RunNow:$false
```
- Jika Anda ingin menentukan folder proyek / hasil:
```
pwsh -ExecutionPolicy Bypass -File .\run_pipeline_complete_with_ui.ps1 -ProjectDir "C:\MyProject" -WorkingFolderName "hasil" -RunNow:$false
```

Apa yang dibuat skrip
- pipeline_run.py — pipeline Python (download data, preprocess, train, forecast, output CSV + report).
- notebooks_html/report_combined_yoy_and_notebook.html — halaman gabungan (image + YoY editor).
- yoy_analysis_with_explanation.html — halaman YoY standalone.
- Struktur folder:
  - data/raw
  - data/processed
  - models
  - notebooks_html
  - results

Membuka UI (direkomendasikan via HTTP lokal)
- Untuk menghindari batasan browser membuka file lokal, jalankan server HTTP sederhana dari folder `hasil/notebooks_html`:
```
cd "C:\Users\<Anda>\Desktop\time-series-forecasting\hasil\notebooks_html"
python -m http.server 8000
```
- Kemudian buka browser ke:
```
http://localhost:8000/report_combined_yoy_and_notebook.html
```
- Standalone YoY: buka file `hasil\yoy_analysis_with_explanation.html` langsung di browser atau dari folder hasil.

Petunjuk singkat penggunaan — Fokus: Analisis Perbandingan Waktu (YoY Editor)
1. Siapkan CSV minimal (kolom `date` dan `value`). Format tanggal disarankan ISO: `YYYY-MM-DD`.
   Contoh isi CSV:
   ```
   date,value
   2019-01-01,100
   2020-01-01,120
   2021-01-01,140
   2022-01-01,150
   2023-01-01,170
   ```
2. Buka halaman YoY editor (gabungan atau standalone).
3. Unggah CSV lewat kontrol "CSV" atau tambahkan data manual per tahun:
   - Masukkan Year, Value, Count lalu klik "Add" / "Tambah".
   - Count berguna jika Anda memasukkan nilai agregat (mis. total) lalu ingin memakai AVG; aturan perhitungan:
     - AVG: jika count>0, value_for_calc = value / count
     - SUM: value_for_calc = value
     - LAST: value_for_calc = value
4. Pilih rentang tahun (From / To) dan mode agregasi (AVG / SUM / LAST).
5. Klik "Generate" → tabel hasil + grafik akan muncul.
6. Optional: edit sel (Year, Value, Count, Diff, %) inline dengan tombol Edit per baris. Jika Anda mengubah Diff/% secara manual, baris ditandai sebagai manual dan tidak akan dihitung ulang otomatis.
7. Untuk menempelkan grafik ke gambar kiri (overlay):
   - Pastikan ada gambar di panel kiri (upload PNG/JPG) — jika tidak ada, fitur Edit PNG akan membuat kanvas putih sebagai dasar.
   - Klik "Edit PNG" pada panel kiri untuk menerapkan overlay.
   - Gunakan "Save Edited" untuk mengunduh gambar hasil overlay.
   - Gunakan "Revert" untuk mengembalikan gambar ke kondisi asli bila tersedia.
8. Ekspor / Simpan:
   - Export CSV / Save CSV: mengekspor tabel saat ini ke file CSV.
   - Download .ipynb (jika tersedia) untuk ringkasan hasil.

Contoh CSV cepat (siap disalin)
```
date,value
2019-01-01,100
2020-01-01,120
2021-01-01,140
2022-01-01,150
2023-01-01,170
```

Troubleshooting singkat
- "No image loaded." — upload file PNG/JPG di panel kiri.
- Hasil kosong / tanda minus pada tabel:
  - Pastikan format tanggal benar (YYYY-MM-DD disarankan).
  - Pastikan rentang tahun mencakup data Anda.
- Jika pipeline Python gagal:
  - Cek apakah Python di PATH dan paket yang dibutuhkan terpasang.
  - Jalankan `python pipeline_run.py` dari folder hasil untuk melihat pesan error lengkap.

Catatan teknis ringkas
- Perubahan penting UI: menambahkan kolom Count yang bisa diedit inline; compute logic memperhitungkan Count untuk mode AVG.
- Edit per baris yang mengubah Diff/% akan menandai baris sebagai manual sehingga tidak akan otomatis dihitung ulang.
- Semua JavaScript UI, HTML dan pipeline Python ditulis oleh skrip PowerShell dan disimpan di folder hasil.

Lisensi & Kontak
- Sesuaikan lisensi sesuai kebutuhan proyek Anda.
- Untuk bantuan lebih lanjut: sertakan screenshot atau lampirkan isi CSV yang dipakai agar saya bisa bantu debugging.

Terakhir: salin seluruh isi README ini dan simpan sebagai `README.md` di folder proyek Anda. Selamat mencoba!

```
<#
run_pipeline_complete_with_ui.ps1 (v63 - full, complete assembly) with editable Count and full Add (Year/Value/Count) support

What changed:
- Both HTML pages (combined and standalone) now allow editing the Count column inline.
- Added "Count" input next to the Add/Manual inputs so Add inserts Year, Value and Count.
- Manual data is stored as { value, count } per year. compute functions treat manual entries as:
    - AVG mode: value_for_calc = (count > 0) ? (value / count) : value
    - SUM mode: value_for_calc = value
    - LAST mode: value_for_calc = value
  This keeps behavior consistent when user supplies aggregated values + counts.
- Editing Count updates internal structures and immediately recalculates Diff/% and chart.
- Edit per-row (existing) now allows editing Count as well; manual overrides for Diff/% are preserved as before.
- All other behavior and layout unchanged.

Usage:
  pwsh -ExecutionPolicy Bypass -File .\run_pipeline_complete_with_ui.ps1
  pwsh -ExecutionPolicy Bypass -File .\run_pipeline_complete_with_ui.ps1 -RunNow:$false
#>

param(
  [string] $ProjectDir = "$HOME\Desktop\time-series-forecasting",
  [string] $WorkingFolderName = "hasil",
  [switch] $RunNow = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dirs {
  param([string]$Base)
  if (-not (Test-Path -LiteralPath $Base)) { New-Item -ItemType Directory -Path $Base -Force | Out-Null }
  $subs = @('data','data\raw','data\processed','models','notebooks_html','results')
  foreach ($s in $subs) {
    $p = Join-Path -Path $Base -ChildPath $s
    if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
  }
}

function Write-FileSafely {
  param(
    [Parameter(Mandatory=$true)][string] $Path,
    [Parameter(Mandatory=$true)][string] $Content,
    [string] $Encoding = "utf8"
  )
  $dir = Split-Path -Path $Path -Parent
  if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  Set-Content -LiteralPath $Path -Value $Content -Encoding $Encoding -Force
  Write-Host "Wrote: $Path" -ForegroundColor Green
}

try {
  $HasilDir = Join-Path -Path $ProjectDir -ChildPath $WorkingFolderName
  Write-Host "ProjectDir: $ProjectDir"
  Write-Host "Working folder (hasil): $HasilDir"
  Ensure-Dirs -Base $HasilDir

  # -----------------------
  # pipeline_run.py (v42) - unchanged
  # -----------------------
  $pyPath = Join-Path $HasilDir "pipeline_run.py"
  $pyContent = @'
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
'@
  Write-FileSafely -Path $pyPath -Content $pyContent -Encoding "utf8"

  # -----------------------
  # Combined UI (report_combined_yoy_and_notebook.html) - UPDATED: Count editable and Add Count input
  # -----------------------
  $combinedPath = Join-Path -Path $HasilDir -ChildPath "notebooks_html/report_combined_yoy_and_notebook.html"
  $combinedContent = @'
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Combined Report — Image + YoY Editor</title>
  <meta name="description" content="Upload image left, create YoY chart right, edit (overlay) chart onto uploaded image and save edited image. If no image present, Edit will create a new base image to draw overlay." />
  <style>
    :root{
      --bg: linear-gradient(180deg,#f8fbff,#f2f7ff);
      --muted: #6b7280;
      --accent: #6366f1;
      --card: #fff;
      --control-gap: 10px;
      --btn-height: 36px;
    }
    html,body{height:100%;margin:0;background:var(--bg);font-family:Inter,system-ui,Roboto,Arial;color:#071032}
    .container{max-width:1200px;margin:18px auto;display:grid;grid-template-columns:640px 1fr;gap:18px}
    .card{background:var(--card);padding:14px;border-radius:12px;box-shadow:0 12px 30px rgba(15,23,42,0.06);border:1px solid rgba(15,23,42,0.04)}
    h2{margin:0 0 6px 0}
    .small{font-size:13px;color:var(--muted);line-height:1.35}
    .img-wrap{height:560px;border-radius:10px;background:#fff;border:1px solid rgba(6,182,212,0.04);overflow:auto;display:flex;align-items:center;justify-content:center;padding:12px}
    .controls{display:flex;flex-wrap:wrap;align-items:center;gap:var(--control-gap);margin-top:12px}
    .file-control{display:flex;align-items:center;gap:8px}
    input[type=file]{height:var(--btn-height)}
    .btn{height:var(--btn-height);padding:0 12px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);background:#fff;cursor:pointer}
    .btn.primary{background:var(--accent);color:#fff;border:none;box-shadow:0 4px 10px rgba(99,102,241,0.12)}
    .btn.ghost{background:#fff}
    .btn-group{display:flex;gap:8px;align-items:center}
    .params{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}
    label{font-size:13px}
    input[type=range]{vertical-align:middle}
    input[type=number]{padding:6px;border-radius:6px;border:1px solid #e6edf7;width:80px}
    select{padding:6px;border-radius:6px;border:1px solid #e6edf7}
    .right-col .card{margin-bottom:12px}
    .chart-canvas{background:#fff;border-radius:8px;padding:8px}
    table{width:100%;border-collapse:collapse;margin-top:12px}
    table th, table td{padding:8px;border-bottom:1px solid #f1f5f9;text-align:left}
    td.editable{background:#fff7e6}
    td.actions{width:160px;text-align:center}
    @media(max-width:980px){
      .container{grid-template-columns:1fr}
      .img-wrap{height:320px}
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
  <div class="container">
    <section class="card">
      <h2>Plot (PNG / JPG)</h2>
      <div class="small">Upload pipeline PNG/JPG. Use "Edit PNG" to apply chart overlay — the left image will be edited in-place. If no image exists, Edit will create a new white canvas and draw the overlay. Use "Revert" to restore uploaded original, "Delete" to remove uploaded/edited image, and "Save Edited" to download the edited image.</div>

      <div class="controls" style="margin-top:16px">
        <div class="file-control">
          <input id="plotImageFile" type="file" accept="image/png,image/jpeg" />
          <div id="leftInfo" style="font-size:13px;color:var(--muted);margin-left:6px">Loaded image: —</div>
        </div>

        <div class="btn-group" style="margin-left:auto">
          <button id="editLeftPng" class="btn primary">Edit PNG</button>
          <button id="revertLeftPng" class="btn">Revert</button>
          <button id="deleteLeftPng" class="btn">Delete</button>
          <button id="saveEditedPng" class="btn">Save Edited</button>
        </div>
      </div>

      <div class="img-wrap" id="leftCanvasWrap" style="margin-top:14px">
        <div class="small">No image loaded.</div>
      </div>

      <div class="params">
        <label>Opacity: <input id="overlayOpacity" type="range" min="0" max="1" step="0.05" value="0.9" /></label>
        <label>Scale (%): <input id="overlayScale" type="number" value="30" />%</label>
        <label>Position:
          <select id="overlayPosition">
            <option value="top-right">Top-right</option>
            <option value="top-left">Top-left</option>
            <option value="bottom-right">Bottom-right</option>
            <option value="bottom-left">Bottom-left</option>
            <option value="center">Center</option>
          </select>
        </label>
      </div>
    </section>

    <aside class="card right-col">
      <h2>Analisis Perbandingan Waktu — YoY Editor</h2>
      <div class="small">Unggah CSV (date,value) atau masukkan data manual. Klik Generate untuk membuat Chart (Chart.js). Setelah generate, gunakan "Edit PNG" untuk menerapkan overlay ke gambar kiri (atau membuat base baru jika belum ada), lalu "Save Edited" untuk menyimpan hasil.</div>

      <div class="card" style="margin-top:12px">
        <label>CSV (date,value): <input id="csvFile" type="file" accept=".csv" /></label>
        <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
          <input id="manualYear" type="number" placeholder="Year" style="width:90px" />
          <input id="manualValue" type="number" placeholder="Value" style="width:120px" />
          <input id="manualCount" type="number" placeholder="Count" style="width:90px" min="0" value="1" />
          <button id="addManualBtn" class="btn">Tambah</button>
        </div>

        <div style="margin-top:12px;display:flex;gap:8px;align-items:center">
          <label>Rentang tahun:</label>
          <input id="yearFrom" type="number" value="2019" style="width:100px" />
          <input id="yearTo" type="number" value="2024" style="width:100px" />
        </div>

        <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
          <label>Mode periode:
            <select id="periodMode"><option value="fullYear">Full year</option><option value="monthDayRange">Month-day range</option><option value="customDates">Custom dates</option></select>
          </label>
          <label>Agregasi:
            <select id="aggMode"><option value="avg">AVG</option><option value="sum">SUM</option><option value="last">LAST</option></select>
          </label>
        </div>

        <div style="margin-top:12px;display:flex;gap:8px">
          <button id="generateBtn" class="btn primary">Generate</button>
          <button id="exportCsvBtn" class="btn">Export CSV</button>
          <button id="downloadIpynb" class="btn">Download .ipynb</button>
        </div>
      </div>

      <div class="card" style="margin-top:12px">
        <strong>Hasil & Visualisasi</strong>
        <div style="margin-top:8px" class="chart-canvas">
          <canvas id="chartCanvas" width="520" height="260"></canvas>
        </div>

        <div style="display:flex;gap:8px;align-items:center;margin-top:8px">
          <label style="font-size:13px"><input id="overlayIfImage" type="checkbox" checked /> Overlay ke left image</label>
          <button id="downloadChartPng" class="btn" style="margin-left:auto">Save Chart PNG</button>
          <button id="saveTableCsv" class="btn" style="margin-left:8px">Save Table CSV</button>
        </div>

        <table id="resultTable" style="margin-top:12px;display:none">
          <thead><tr><th>Year</th><th>Value</th><th>Count</th><th>Diff</th><th>%</th><th>Aksi</th></tr></thead>
          <tbody></tbody>
        </table>

        <div style="margin-top:12px">
          <strong>Notebook preview</strong>
          <pre id="notebookPreview" style="height:120px;background:#0b1220;color:#e6eef8;padding:8px;border-radius:6px;overflow:auto">Notebook preview will appear here.</pre>
        </div>
      </div>
    </aside>
  </div>

<script>
/* --- Helpers --- */
function parseCSV(text){
  const lines = text.replace(/\r\n/g,'\n').split('\n').filter(l=>l.trim()!=='');
  if(!lines.length) return {header:[], rows:[]};
  const header = lines[0].split(',').map(h=>h.trim());
  const rows = lines.slice(1).map(l=>l.split(',').map(c=>c.trim()));
  return {header, rows};
}
function toDate(s){
  if(!s) return null;
  const d = Date.parse(s);
  if(!isNaN(d)) return new Date(d);
  const m = s && s.match(/^(\d{1,2})[\/\-\.\s](\d{1,2})[\/\-\.\s](\d{2,4})$/);
  if(m){ let a=parseInt(m[1],10), b=parseInt(m[2],10)-1, c=parseInt(m[3],10); if(c<100) c+=2000; return new Date(c,b,a); }
  return null;
}
function fmtNum(v){ return (v===null||v===undefined) ? "-" : Number(v).toLocaleString('en-US',{maximumFractionDigits:2}); }

/* --- Elements --- */
const plotImageFile = document.getElementById('plotImageFile');
const leftCanvasWrap = document.getElementById('leftCanvasWrap');
const leftInfo = document.getElementById('leftInfo');
const editLeftPng = document.getElementById('editLeftPng');
const revertLeftPng = document.getElementById('revertLeftPng');
const deleteLeftPng = document.getElementById('deleteLeftPng');
const saveEditedPng = document.getElementById('saveEditedPng');

const csvFile = document.getElementById('csvFile');
const manualYear = document.getElementById('manualYear');
const manualValue = document.getElementById('manualValue');
const manualCount = document.getElementById('manualCount');
const addManualBtn = document.getElementById('addManualBtn');

const yearFrom = document.getElementById('yearFrom');
const yearTo = document.getElementById('yearTo');
const periodMode = document.getElementById('periodMode');
const aggMode = document.getElementById('aggMode');
const generateBtn = document.getElementById('generateBtn');
const exportCsvBtn = document.getElementById('exportCsvBtn');
const downloadIpynbBtn = document.getElementById('downloadIpynb');

const chartCanvas = document.getElementById('chartCanvas');
const downloadChartPng = document.getElementById('downloadChartPng');
const saveTableCsv = document.getElementById('saveTableCsv');
const overlayIfImage = document.getElementById('overlayIfImage');
const overlayOpacity = document.getElementById('overlayOpacity');
const overlayScale = document.getElementById('overlayScale');
const overlayPosition = document.getElementById('overlayPosition');

const resultTable = document.getElementById('resultTable');
const resultTbody = resultTable.querySelector('tbody');
const notebookPreview = document.getElementById('notebookPreview');

let rawData = []; // array of {date, value}
let manualData = {}; // year => { value: number, count: number }
let chart = null;

/* LEFT: image handlers unchanged */
plotImageFile.addEventListener('change', e=>{
  const f = e.target.files && e.target.files[0];
  if(!f) return;
  leftCanvasWrap.innerHTML = '';
  leftInfo.textContent = "Loading image...";
  const reader = new FileReader();
  reader.onload = function(ev){
    const dataUrl = ev.target.result;
    const img = document.createElement('img');
    img.style.maxWidth = "100%";
    img.style.maxHeight = "100%";
    img.src = dataUrl;
    img.dataset.original = dataUrl;
    img.dataset.filename = f.name;
    delete img.dataset.edited;
    leftCanvasWrap.appendChild(img);
    leftInfo.textContent = `Loaded image: ${f.name}`;
  };
  reader.readAsDataURL(f);
});
deleteLeftPng.addEventListener('click', ()=>{ leftCanvasWrap.innerHTML = '<div class="small">No image loaded.</div>'; leftInfo.textContent = "Loaded image: —"; });
revertLeftPng.addEventListener('click', ()=>{ const img = leftCanvasWrap.querySelector('img'); if(!img){ alert('No image loaded'); return; } if(img.dataset && img.dataset.original){ img.src = img.dataset.original; leftInfo.textContent = `Reverted to original: ${img.dataset.filename || 'uploaded'}`; delete img.dataset.edited; } else { alert('Original image not available to revert.'); }});
saveEditedPng.addEventListener('click', ()=>{ const img = leftCanvasWrap.querySelector('img'); if(!img || !img.src){ alert('No image loaded'); return; } const dataUrl = img.dataset.edited || img.src; const originalName = (img.dataset && img.dataset.filename) ? img.dataset.filename.replace(/\.[^/.]+$/, "") : "left_image"; const suggested = `${originalName}-edited.png`; const a = document.createElement('a'); a.href = dataUrl; a.download = suggested; document.body.appendChild(a); a.click(); a.remove(); });

/* CSV import -> rawData unchanged */
csvFile.addEventListener('change', e=>{
  const f = e.target.files && e.target.files[0];
  if(!f) return;
  const r = new FileReader();
  r.onload = ()=>{
    const parsed = parseCSV(r.result);
    const header = parsed.header, rows = parsed.rows;
    let dateIdx=-1, valIdx=-1;
    for(let i=0;i<header.length;i++){
      let dcount=0,ncount=0;
      for(let j=0;j<Math.min(rows.length,80); j++){
        const v = rows[j][i]||"";
        if(toDate(v)) dcount++;
        if(v!=="" && !isNaN(parseFloat(v))) ncount++;
      }
      if(dcount >= Math.max(1, Math.floor(rows.length*0.25))) dateIdx = i;
      if(ncount >= Math.max(1, Math.floor(rows.length*0.25))) valIdx = i;
    }
    if(dateIdx === -1) dateIdx = 0;
    if(valIdx === -1) valIdx = (header.length>1?1:0);
    rawData = [];
    rows.forEach(rw=>{
      const d = toDate(rw[dateIdx]);
      const v = parseFloat(rw[valIdx]);
      if(d && !isNaN(v)) rawData.push({date:d, value:v});
    });
    rawData.sort((a,b)=>a.date - b.date);
    notebookPreview.textContent = rawData.slice(0,50).map(r=>`${r.date.toISOString().slice(0,10)},${r.value}`).join('\n') || "No valid rows";
    alert(`CSV loaded — ${rawData.length} valid rows`);
  };
  r.readAsText(f);
});

/* Add manual row (Year/Value/Count) -> store in manualData as {value,count} */
addManualBtn.addEventListener('click', ()=>{
  const y = parseInt(manualYear.value,10);
  const v = manualValue.value === '' ? null : parseFloat(manualValue.value);
  const c = manualCount.value === '' ? 1 : parseInt(manualCount.value,10);
  if(isNaN(y) || (v===null || isNaN(v)) || isNaN(c) || c < 0){ alert('Masukkan Year, Value, dan Count yang valid'); return; }
  manualData[y] = { value: v, count: c };
  manualYear.value=''; manualValue.value=''; manualCount.value='1';
  alert('Manual row added');
});

/* Compute YoY now respects manualData entries with counts:
   - if useManual (manualData exists & rawData empty), for each year:
     - if agg == "avg": value_for_series = manual.value / manual.count (if count>0)
     - if agg == "sum": value_for_series = manual.value
     - if agg == "last": value_for_series = manual.value
   The "count" column is used as informational / to compute AVG when user provides aggregated value.
*/
function computeYoY(fromYear, toYear, mode, agg){
  const results = [];
  const useManual = (Object.keys(manualData).length > 0 && rawData.length === 0);
  for(let y=fromYear;y<=toYear;y++){
    let start = new Date(y,0,1), end = new Date(y,11,31,23,59,59);
    if(mode === 'monthDayRange'){
      const s = document.getElementById('startMMDD') ? document.getElementById('startMMDD').value || '01-01' : '01-01';
      const e = document.getElementById('endMMDD') ? document.getElementById('endMMDD').value || '12-31' : '12-31';
      const sp = s.split('-'), ep = e.split('-');
      start = new Date(y, parseInt(sp[0],10)-1, parseInt(sp[1],10));
      end = new Date(y, parseInt(ep[0],10)-1, parseInt(ep[1],10), 23,59,59);
    } else if(mode === 'customDates'){
      const stElem = document.getElementById('startTemplate'), enElem = document.getElementById('endTemplate');
      const st = stElem ? stElem.value : null;
      const en = enElem ? enElem.value : null;
      if(st && en){
        const sdate = new Date(st), edate = new Date(en);
        start = new Date(y, sdate.getMonth(), sdate.getDate());
        end = new Date(y, edate.getMonth(), edate.getDate(), 23,59,59);
      }
    }
    let vals = [];
    let count = 0;
    if(useManual){
      const entry = manualData[y];
      if(entry !== undefined && entry.value != null){
        // interpret manual entry according to agg
        if(agg === 'sum' || agg === 'last'){
          vals = [entry.value];
          count = entry.count || 1;
        } else { // avg
          // if user provided aggregated sum in entry.value with count entry.count,
          // compute average value_for_series = entry.value / count
          if(entry.count && entry.count > 0){
            vals = [entry.value / entry.count];
            count = entry.count;
          } else {
            vals = [entry.value];
            count = 1;
          }
        }
      } else {
        vals = [];
        count = 0;
      }
    } else {
      const filtered = rawData.filter(r=> r.date>=start && r.date<=end).map(r=>r.value);
      vals = filtered;
      count = filtered.length;
    }
    let value = null;
    if(vals.length === 0) value = null;
    else if(agg === 'sum') value = vals.reduce((a,b)=>a+b,0);
    else if(agg === 'last') value = vals[vals.length-1];
    else value = vals.reduce((a,b)=>a+b,0)/vals.length;
    results.push({year:y, value:value, count:count});
  }
  for(let i=0;i<results.length;i++){
    const cur=results[i], prev=i>0?results[i-1]:null;
    if(prev && cur.value!=null && prev.value!=null){ cur.diff = cur.value - prev.value; cur.pct = prev.value!==0 ? (cur.diff/Math.abs(prev.value))*100 : null; } else { cur.diff=null; cur.pct=null; }
  }
  return results;
}

/* Render results: includes Count as editable column and Edit/Save per-row plus Hapus */
function renderResults(results){
  resultTbody.innerHTML = '';
  const labels = [], data = [];
  results.forEach(r=>{
    const tr = document.createElement('tr');

    const yearTd = document.createElement('td'); yearTd.className = 'year'; yearTd.contentEditable = 'false'; yearTd.innerText = r.year;
    const valueTd = document.createElement('td'); valueTd.className = 'value'; valueTd.contentEditable = 'false'; valueTd.innerText = r.value===null? '' : r.value;
    const countTd = document.createElement('td'); countTd.className = 'count'; countTd.contentEditable = 'false'; countTd.innerText = (r.count===null || r.count===undefined) ? '' : String(r.count);

    const diffTd = document.createElement('td'); diffTd.className = 'diff'; diffTd.contentEditable = 'false'; diffTd.innerText = r.diff===null? '-' : fmtNum(r.diff);
    const pctTd = document.createElement('td'); pctTd.className = 'pct'; pctTd.contentEditable = 'false'; pctTd.innerText = r.pct===null? '-' : (r.pct.toFixed(2)+'%');

    const actionTd = document.createElement('td'); actionTd.className='actions';
    const editBtn = document.createElement('button'); editBtn.className='btn'; editBtn.innerText='Edit';
    const delBtn = document.createElement('button'); delBtn.className='btn'; delBtn.style.marginLeft='6px'; delBtn.innerText='Hapus';
    actionTd.appendChild(editBtn); actionTd.appendChild(delBtn);

    // Edit toggle
    editBtn.addEventListener('click', ()=>{
      const editing = tr.dataset.editing === 'true';
      if(!editing){
        tr.dataset.editing = 'true';
        tr.dataset.origDiff = diffTd.innerText;
        tr.dataset.origPct = pctTd.innerText;
        yearTd.contentEditable = 'true';
        valueTd.contentEditable = 'true';
        countTd.contentEditable = 'true';
        diffTd.contentEditable = 'true';
        pctTd.contentEditable = 'true';
        editBtn.innerText = 'Save';
        valueTd.focus();
      } else {
        // Save
        yearTd.contentEditable = 'false';
        valueTd.contentEditable = 'false';
        countTd.contentEditable = 'false';
        diffTd.contentEditable = 'false';
        pctTd.contentEditable = 'false';
        // If user changed diff/pct manually, set manual flag
        if(diffTd.innerText.trim() !== (tr.dataset.origDiff || '') || pctTd.innerText.trim() !== (tr.dataset.origPct || '')){
          tr.dataset.manual = 'true';
        }
        delete tr.dataset.editing;
        editBtn.innerText = 'Edit';
        // Update manualData mapping if rawData not used
        const y = parseInt(yearTd.innerText.trim(),10);
        const vtxt = valueTd.innerText.trim();
        const ctxt = countTd.innerText.trim();
        const v = vtxt === '' ? null : parseFloat(vtxt);
        const c = ctxt === '' ? 0 : parseInt(ctxt,10);
        if(!isNaN(y)){
          if(v === null || isNaN(v)){
            delete manualData[y];
          } else {
            manualData[y] = { value: v, count: (isNaN(c)? 0 : c) };
          }
        }
        recalcFromTable();
      }
    });

    // Delete
    delBtn.addEventListener('click', ()=>{
      const y = parseInt(yearTd.innerText.trim(),10);
      if(!isNaN(y) && manualData[y] !== undefined){
        delete manualData[y];
      }
      tr.remove();
      recalcFromTable();
    });

    // Blur handlers to update manualData/count on inline edits (when not in edit mode this won't trigger changes)
    valueTd.addEventListener('blur', ()=> {
      const y = parseInt(yearTd.innerText.trim(),10);
      const vtxt = valueTd.innerText.trim();
      const v = vtxt === '' ? null : parseFloat(vtxt);
      if(!isNaN(y)){
        const existing = manualData[y] || { value: null, count: 0 };
        if(v === null || isNaN(v)){
          delete manualData[y];
        } else {
          existing.value = v;
          existing.count = existing.count || 1;
          manualData[y] = existing;
        }
      }
      recalcFromTable();
    });
    countTd.addEventListener('blur', ()=> {
      const y = parseInt(yearTd.innerText.trim(),10);
      const ctxt = countTd.innerText.trim();
      const c = ctxt === '' ? 0 : parseInt(ctxt,10);
      if(!isNaN(y)){
        const existing = manualData[y] || { value: null, count: 0 };
        existing.count = isNaN(c) ? 0 : c;
        if(existing.value === null || existing.value === undefined){
          // keep but count updated
          manualData[y] = existing;
        } else {
          manualData[y] = existing;
        }
      }
      recalcFromTable();
    });
    yearTd.addEventListener('blur', ()=> recalcFromTable());

    tr.appendChild(yearTd);
    tr.appendChild(valueTd);
    tr.appendChild(countTd);
    tr.appendChild(diffTd);
    tr.appendChild(pctTd);
    tr.appendChild(actionTd);

    resultTbody.appendChild(tr);

    labels.push(String(r.year)); data.push(r.value===null?NaN:r.value);
  });

  resultTable.style.display = results.length ? 'table' : 'none';
  if(chart) try{ chart.destroy(); }catch(e){}
  const ctx = chartCanvas.getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets:[{ label:'Value', data, borderColor:'rgba(99,102,241,1)', backgroundColor:'rgba(99,102,241,0.12)', fill:true, tension:0.25 }] },
    options: { responsive:true, plugins:{legend:{display:false}}, scales:{ y:{beginAtZero:false} } }
  });
  notebookPreview.textContent = "YoY summary\n" + results.map(r=>`${r.year}, value=${r.value===null?'-':r.value}, count=${r.count}`).join('\n');
}

/* recalcFromTable updated to respect manualData and editable Count */
function recalcFromTable(){
  const trs = Array.from(resultTbody.querySelectorAll('tr'));
  // Build array of rows with year/value/count
  const rows = trs.map(tr=>{
    const y = parseInt(tr.querySelector('.year').innerText.trim(),10);
    const vtxt = tr.querySelector('.value').innerText.trim();
    const ctxt = tr.querySelector('.count').innerText.trim();
    const v = vtxt === '' ? null : parseFloat(vtxt);
    const c = ctxt === '' ? 0 : parseInt(ctxt,10);
    return { year: y, value: (isNaN(v)? null: v), count: (isNaN(c)? 0: c), tr: tr };
  }).filter(r=> !isNaN(r.year));
  rows.sort((a,b)=> a.year - b.year);

  // Update manualData mapping from table where relevant:
  rows.forEach(r=>{
    if(r.value !== null){
      manualData[r.year] = { value: r.value, count: r.count || 0 };
    } else {
      if(manualData[r.year] && (manualData[r.year].value === null || manualData[r.year].value === undefined)) {
        delete manualData[r.year];
      }
    }
  });

  // Now compute diffs/pct skipping rows flagged manual (data-manual="true" on tr)
  for(let i=0;i<rows.length;i++){
    const cur = rows[i], prev = i>0?rows[i-1]:null;
    const tr = cur.tr;
    const diffCell = tr.querySelector('.diff');
    const pctCell = tr.querySelector('.pct');
    if(prev && cur.value!=null && prev.value!=null){
      const diff = cur.value - prev.value;
      const pct = prev.value !== 0 ? (diff/Math.abs(prev.value))*100 : null;
      if(tr.dataset.manual !== 'true'){
        diffCell.innerText = fmtNum(diff);
        pctCell.innerText = pct===null? '-' : pct.toFixed(2) + '%';
      }
    } else {
      if(tr.dataset.manual !== 'true'){
        diffCell.innerText = '-';
        pctCell.innerText = '-';
      }
    }
  }

  // update chart (use displayed value)
  const labels = rows.map(r=>String(r.year));
  const data = rows.map(r=> r.value===null?NaN:r.value);
  if(chart) chart.destroy();
  chart = new Chart(chartCanvas.getContext('2d'), {
    type: 'line',
    data: { labels, datasets:[{ label:'Value', data, borderColor:'rgba(99,102,241,1)', backgroundColor:'rgba(99,102,241,0.12)', fill:true, tension:0.25 }] },
    options: { responsive:true, plugins:{legend:{display:false}} }
  });
}

/* Generate button handler unchanged except computeYoY now returns count */
generateBtn.addEventListener('click', ()=>{
  const from = parseInt(yearFrom.value,10), to = parseInt(yearTo.value,10);
  if(isNaN(from) || isNaN(to) || from>to){ alert("Invalid year range"); return; }
  const agg = aggMode.value;
  const results = computeYoY(from, to, periodMode.value, agg);
  renderResults(results);
});

/* download chart PNG */
downloadChartPng.addEventListener('click', ()=>{
  if(!chart){ alert("No chart"); return; }
  try { const url = chart.toBase64Image(); const a = document.createElement('a'); a.href = url; a.download = "combined_yoy_chart.png"; document.body.appendChild(a); a.click(); a.remove(); } catch(e){ alert("Failed to export chart PNG"); }
});

/* export current table rows */
exportCsvBtn.addEventListener('click', ()=>{
  const trs = Array.from(resultTbody.querySelectorAll('tr'));
  if(!trs.length){ alert('No results to export'); return; }
  const rows = [['Year','Value','Count','Diff','Pct']];
  trs.forEach(tr=>{
    const year = tr.querySelector('.year').innerText.trim();
    const val = tr.querySelector('.value').innerText.trim();
    const cnt = tr.querySelector('.count').innerText.trim();
    const diff = tr.querySelector('.diff').innerText.trim();
    const pct = tr.querySelector('.pct').innerText.trim();
    rows.push([year, val, cnt, diff, pct]);
  });
  const csv = rows.map(r=> r.join(',')).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'})); a.download = 'yoy_results_export.csv'; document.body.appendChild(a); a.click(); a.remove();
});

/* Save Table CSV explicit */
saveTableCsv.addEventListener('click', ()=>{
  const trs = Array.from(resultTbody.querySelectorAll('tr'));
  if(!trs.length){ alert('No rows to save'); return; }
  recalcFromTable();
  const rows = [['Year','Value','Count','Diff','Pct']];
  trs.forEach(tr=>{
    const year = tr.querySelector('.year').innerText.trim();
    const val = tr.querySelector('.value').innerText.trim();
    const cnt = tr.querySelector('.count').innerText.trim();
    const diff = tr.querySelector('.diff').innerText.trim();
    const pct = tr.querySelector('.pct').innerText.trim();
    rows.push([year, val, cnt, diff, pct]);
  });
  const csv = rows.map(r=> r.join(',')).join('\n');
  const ts = new Date().toISOString().replace(/[:\-]/g,'').split('.')[0];
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'})); a.download = `yoy_results_saved_${ts}.csv`; document.body.appendChild(a); a.click(); a.remove();
});
</script>
</body>
</html>
'@

  Write-FileSafely -Path $combinedPath -Content $combinedContent -Encoding "utf8"

  # -----------------------
  # Standalone YoY editor - UPDATED: Count editable, Add supports Count, Edit includes Count
  # -----------------------
  $htmlYoYPath = Join-Path -Path $HasilDir -ChildPath "yoy_analysis_with_explanation.html"
  $htmlYoYContent = @'
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>YoY Standalone Editor — Edit / Hapus / Simpan CSV</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root{ --bg:#f4f8ff; --muted:#6b7280; --accent:#6366f1 }
    body{font-family:Inter,Arial;margin:18px;background:linear-gradient(180deg,#f8fbff,#f2f7ff);color:#071032}
    .card{background:#fff;padding:18px;border-radius:10px;box-shadow:0 10px 30px rgba(15,23,42,0.06);max-width:1100px;margin:0 auto}
    .row{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
    input,select,button{padding:8px;border-radius:8px;border:1px solid #e6edf7}
    button.primary{background:#6366f1;color:#fff;border:none}
    table{width:100%;border-collapse:collapse;margin-top:12px}
    th,td{padding:8px;border:1px solid #eef2ff;text-align:left}
    td.editable{background:#fff7e6}
    .hint{font-size:13px;color:#6b7280}
    .small-btn{padding:6px 8px;border-radius:6px}
    .actions{width:160px;text-align:center}
    @media(max-width:720px){ .row{flex-direction:column;align-items:stretch} }
  </style>
</head>
<body>
  <div class="card">
    <h2>YoY Standalone Editor — Edit / Hapus / Simpan CSV</h2>
    <p class="hint">Impor CSV (date,value) atau tambahkan data manual (Year, Value, Count). Setelah Generate, Anda dapat mengedit Year/Value/Count/Diff/% inline, menghapus baris, dan mengekspor seluruh tabel ke CSV.</p>

    <div class="row">
      <label>CSV: <input id="csvFile" type="file" accept=".csv" /></label>
      <div style="display:flex;gap:8px;margin-left:8px">
        <input id="yearInput" type="number" placeholder="Year" style="width:110px" />
        <input id="valueInput" type="number" placeholder="Value" style="width:140px" />
        <input id="countInput" type="number" placeholder="Count" style="width:100px" min="0" value="1" />
        <button id="addBtn" class="small-btn">Add</button>
      </div>
      <div style="margin-left:auto;display:flex;gap:8px">
        <label>From <input id="fromYear" type="number" value="2019" style="width:100px;margin-left:6px"/></label>
        <label>To <input id="toYear" type="number" value="2024" style="width:100px;margin-left:6px"/></label>
        <select id="aggMode" style="margin-left:6px"><option value="avg">AVG</option><option value="sum">SUM</option><option value="last">LAST</option></select>
        <button id="generateBtn" class="small-btn">Generate</button>
        <button id="saveCsvBtn" class="small-btn">Save CSV</button>
        <button id="clearBtn" class="small-btn" style="background:#ef4444;color:#fff">Clear</button>
      </div>
    </div>

    <div class="table-scroll">
      <table id="dataTable"><thead><tr><th>Year</th><th>Value</th><th>Count</th><th>Diff</th><th>%</th><th>Aksi</th></tr></thead><tbody></tbody></table>
    </div>

    <div style="margin-top:18px"><canvas id="chart" width="1000" height="360" style="background:#fff;border-radius:6px"></canvas></div>
  </div>

<script>
function parseCSV(text){
  const lines = text.replace(/\r\n/g,"\\n").split("\\n").map(l=>l.trim()).filter(l=>l!=="");
  if(!lines.length) return [];
  const first = lines[0].split(",").map(s=>s.trim());
  const header = first.some(h=>isNaN(parseInt(h,10)));
  const rows = header ? lines.slice(1) : lines;
  const out = [];
  for(const r of rows){
    const cols = r.split(",").map(c=>c.trim());
    if(cols.length < 2) continue;
    const y = parseInt(cols[0],10);
    const v = cols[1]==='' ? null : parseFloat(cols[1]);
    const c = cols.length > 2 ? (cols[2]===''?0:parseInt(cols[2],10)) : 1;
    if(!isNaN(y)) out.push({year:y, value:(v===null||isNaN(v))? null : v, count: isNaN(c)? 0 : c});
  }
  return out;
}
function fmtNum(v){ return (v===null||v===undefined) ? "-" : Number(v).toLocaleString('en-US',{maximumFractionDigits:2}); }

const csvFile = document.getElementById('csvFile');
const addBtn = document.getElementById('addBtn');
const yearInput = document.getElementById('yearInput');
const valueInput = document.getElementById('valueInput');
const countInput = document.getElementById('countInput');
const fromYear = document.getElementById('fromYear');
const toYear = document.getElementById('toYear');
const aggMode = document.getElementById('aggMode');
const generateBtn = document.getElementById('generateBtn');
const saveCsvBtn = document.getElementById('saveCsvBtn');
const clearBtn = document.getElementById('clearBtn');

const tbody = document.querySelector('#dataTable tbody');
const chartEl = document.getElementById('chart');
let chartInstance = null;
let tableRows = []; // {year, value, count, diff, pct, manual}

/* CSV import */
csvFile.addEventListener('change', e=>{
  const f = e.target.files && e.target.files[0]; if(!f) return;
  const r = new FileReader(); r.onload = ()=>{
    const parsed = parseCSV(r.result);
    const map = Object.fromEntries(tableRows.map(x=>[x.year, {value:x.value, count:x.count||0}]));
    parsed.forEach(p=> map[p.year] = { value: p.value, count: p.count });
    tableRows = Object.keys(map).map(k=>({year:parseInt(k,10), value: map[k].value, count: map[k].count || 0, diff:null, pct:null, manual:false}));
    tableRows.sort((a,b)=>a.year-b.year);
    renderTable();
    recalcStandalone();
  }; r.readAsText(f);
});

/* Add manual row (Year, Value, Count) */
addBtn.addEventListener('click', ()=>{
  const y = parseInt(yearInput.value,10);
  const v = valueInput.value === '' ? null : parseFloat(valueInput.value);
  const c = countInput.value === '' ? 0 : parseInt(countInput.value,10);
  if(isNaN(y) || (v===null || isNaN(v)) || isNaN(c) || c < 0){ alert('Masukkan Year, Value, dan Count yang valid'); return; }
  // replace existing year if present
  tableRows = tableRows.filter(r=> r.year !== y);
  tableRows.push({year: y, value: v, count: c, diff: null, pct: null, manual: false});
  tableRows.sort((a,b)=> a.year - b.year);
  renderTable();
  recalcStandalone();
  yearInput.value=''; valueInput.value=''; countInput.value='1';
});

/* Render table with Edit/Hapus and Count column editable when editing */
function renderTable(){
  tbody.innerHTML = '';
  tableRows.forEach(r=>{
    const tr = document.createElement('tr');
    tr.dataset.year = r.year;

    const tdY = document.createElement('td'); tdY.className='year'; tdY.contentEditable='false'; tdY.innerText = r.year;
    const tdV = document.createElement('td'); tdV.className='value'; tdV.contentEditable='false'; tdV.innerText = r.value===null? '' : r.value;
    const tdC = document.createElement('td'); tdC.className='count'; tdC.contentEditable='false'; tdC.innerText = r.count===undefined? '' : String(r.count);
    const tdCountInfo = tdC;
    const tdCountHidden = null;

    const tdCnt = tdC;
    const tdDiff = document.createElement('td'); tdDiff.className='diff'; tdDiff.contentEditable='false'; tdDiff.innerText = r.diff===null? '-' : fmtNum(r.diff);
    const tdPct = document.createElement('td'); tdPct.className='pct'; tdPct.contentEditable='false'; tdPct.innerText = r.pct===null? '-' : (r.pct.toFixed(2)+'%');

    const tdAksi = document.createElement('td'); tdAksi.className='actions';
    const editBtn = document.createElement('button'); editBtn.className='small-btn'; editBtn.innerText='Edit';
    const delBtn = document.createElement('button'); delBtn.className='small-btn'; delBtn.innerText='Hapus'; delBtn.style.marginLeft='6px';

    tdAksi.appendChild(editBtn); tdAksi.appendChild(delBtn);

    // Edit toggle: enables editing Year, Value, Count, Diff, Pct
    editBtn.addEventListener('click', ()=>{
      const editing = tr.dataset.editing === 'true';
      if(!editing){
        tr.dataset.editing = 'true';
        tr.dataset.origDiff = tdDiff.innerText;
        tr.dataset.origPct = tdPct.innerText;
        tdY.contentEditable='true'; tdV.contentEditable='true'; tdCnt.contentEditable='true'; tdDiff.contentEditable='true'; tdPct.contentEditable='true';
        editBtn.innerText='Save';
        tdV.focus();
      } else {
        // Save
        tdY.contentEditable='false'; tdV.contentEditable='false'; tdCnt.contentEditable='false'; tdDiff.contentEditable='false'; tdPct.contentEditable='false';
        // if diff/pct changed -> mark manual
        if(tdDiff.innerText.trim() !== (tr.dataset.origDiff || '') || tdPct.innerText.trim() !== (tr.dataset.origPct || '')){
          tr.dataset.manual = 'true';
        }
        delete tr.dataset.editing;
        editBtn.innerText='Edit';
        // update tableRows structure based on edited cells
        const ny = parseInt(tdY.innerText.trim(),10);
        const nv = tdV.innerText.trim() === '' ? null : parseFloat(tdV.innerText.trim());
        const nc = tdCnt.innerText.trim() === '' ? 0 : parseInt(tdCnt.innerText.trim(),10);
        // replace or update row in tableRows
        tableRows = tableRows.filter(x=> x.year !== r.year && x.year !== ny); // remove old and any same-year duplicates
        tableRows.push({ year: ny, value: nv, count: isNaN(nc)?0:nc, diff: tdDiff.innerText.trim() === '-' ? null : parseFloat(tdDiff.innerText.replace(/,/g,'')), pct: tdPct.innerText.trim().endsWith('%') ? parseFloat(tdPct.innerText.trim().replace('%','')) : (tdPct.innerText.trim()==='-'?null:parseFloat(tdPct.innerText)), manual: (tr.dataset.manual === 'true')});
        tableRows.sort((a,b)=> a.year - b.year);
        renderTable();
        recalcStandalone();
      }
    });

    delBtn.addEventListener('click', ()=>{
      tableRows = tableRows.filter(x=> x.year !== r.year);
      renderTable();
      if(chartInstance) chartInstance.destroy();
    });

    tdY.addEventListener('blur', ()=> recalcStandalone());
    tdV.addEventListener('blur', ()=> recalcStandalone());
    tdCnt.addEventListener('blur', ()=> recalcStandalone());

    tr.appendChild(tdY); tr.appendChild(tdV); tr.appendChild(tdCnt); tr.appendChild(tdDiff); tr.appendChild(tdPct); tr.appendChild(tdAksi);
    tbody.appendChild(tr);
  });
}

/* recalcStandalone: compute diffs/pct from current table rows, skip manual rows */
function recalcStandalone(){
  const trs = Array.from(tbody.querySelectorAll('tr'));
  const rows = trs.map(tr=> {
    const y = parseInt(tr.querySelector('.year').innerText.trim(),10);
    const vtxt = tr.querySelector('.value').innerText.trim();
    const v = vtxt === '' ? null : parseFloat(vtxt);
    const ctxt = tr.querySelector('.count').innerText.trim();
    const c = ctxt === '' ? 0 : parseInt(ctxt,10);
    return { year: y, value: isNaN(v)? null : v, count: isNaN(c)? 0 : c, tr: tr };
  }).filter(r=> !isNaN(r.year));
  rows.sort((a,b)=> a.year - b.year);

  // Update underlying tableRows array to reflect edited values and counts
  tableRows = rows.map(r=> ({ year: r.year, value: r.value, count: r.count, diff: null, pct: null, manual: (r.tr.dataset.manual === 'true') }));

  for(let i=0;i<rows.length;i++){
    const cur = rows[i], prev = i>0?rows[i-1]:null;
    const tr = cur.tr;
    const diffCell = tr.querySelector('.diff');
    const pctCell = tr.querySelector('.pct');
    if(prev && cur.value != null && prev.value != null){
      const diff = cur.value - prev.value;
      const pct = prev.value !== 0 ? (diff/Math.abs(prev.value))*100 : null;
      if(tr.dataset.manual !== 'true'){
        diffCell.innerText = fmtNum(diff);
        pctCell.innerText = pct===null? '-' : pct.toFixed(2) + '%';
      }
    } else {
      if(tr.dataset.manual !== 'true'){
        diffCell.innerText = '-';
        pctCell.innerText = '-';
      }
    }
  }
  // update chart
  const labels = tableRows.map(r=> String(r.year));
  const data = tableRows.map(r=> r.value === null ? NaN : r.value);
  if(chartInstance) chartInstance.destroy();
  chartInstance = new Chart(chartEl.getContext('2d'), { type:'line', data:{ labels, datasets:[{ label:'Value', data, borderColor:'rgba(99,102,241,1)', backgroundColor:'rgba(99,102,241,0.12)', fill:true }]}, options:{responsive:true, plugins:{legend:{display:false}}}});
}

/* Generate: compute results using logic similar to combined compute (but using tableRows header if available) */
generateBtn.addEventListener('click', ()=>{
  const from = parseInt(fromYear.value,10); const to = parseInt(toYear.value,10);
  if(isNaN(from) || isNaN(to) || from>to){ alert('Rentang tahun tidak valid'); return; }
  const agg = aggMode.value || 'avg';
  // Build a map from current tableRows (including manual additions)
  const map = Object.fromEntries(tableRows.map(r=>[r.year, { value: r.value, count: r.count || 0 }]));
  const results = [];
  for(let y=from;y<=to;y++){
    const entry = map[y];
    let vals = [];
    let count = 0;
    if(entry && entry.value != null){
      if(agg === 'avg'){
        if(entry.count && entry.count > 0){
          vals = [entry.value / entry.count];
          count = entry.count;
        } else {
          vals = [entry.value];
          count = 1;
        }
      } else {
        vals = [entry.value];
        count = entry.count || 1;
      }
    } else {
      vals = [];
      count = 0;
    }
    let value = null;
    if(vals.length === 0) value = null;
    else if(agg==='sum') value = vals.reduce((a,b)=>a+b,0);
    else if(agg==='last') value = vals[vals.length-1];
    else value = vals.reduce((a,b)=>a+b,0)/vals.length;
    results.push({ year: y, value: value, count: count });
  }
  for(let i=0;i<results.length;i++){
    const cur = results[i], prev = i>0?results[i-1]:null;
    if(prev && cur.value != null && prev.value != null){
      cur.diff = cur.value - prev.value;
      cur.pct = prev.value !== 0 ? (cur.diff/Math.abs(prev.value))*100 : null;
    } else { cur.diff = null; cur.pct = null; }
  }
  // set tableRows and render
  tableRows = results.map(r=>({ year: r.year, value: r.value, count: r.count, diff: r.diff, pct: r.pct, manual: false }));
  renderTable();
  recalcStandalone();
});

/* Save CSV - export current table rows */
saveCsvBtn.addEventListener('click', ()=>{
  const trs = Array.from(tbody.querySelectorAll('tr'));
  if(!trs.length){ alert('No rows to export'); return; }
  recalcStandalone();
  const rows = [['Year','Value','Count','Diff','Pct']];
  trs.forEach(tr=>{
    const year = tr.querySelector('.year').innerText.trim();
    const val = tr.querySelector('.value').innerText.trim();
    const cnt = tr.querySelector('.count').innerText.trim();
    const diff = tr.querySelector('.diff').innerText.trim();
    const pct = tr.querySelector('.pct').innerText.trim();
    rows.push([year, val, cnt, diff, pct]);
  });
  const csv = rows.map(r=> r.join(',')).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'})); const ts = new Date().toISOString().replace(/[:\-]/g,'').split('.')[0]; a.download = `yoy_standalone_${ts}.csv`; document.body.appendChild(a); a.click(); a.remove();
});

/* Clear */
clearBtn.addEventListener('click', ()=>{ if(confirm('Hapus semua baris?')){ tableRows = []; renderTable(); if(chartInstance) chartInstance.destroy(); } });

renderTable();
</script>
</body>
</html>
'@

  Write-FileSafely -Path $htmlYoYPath -Content $htmlYoYContent -Encoding "utf8"

  Write-Host "`nAll write operations completed." -ForegroundColor Cyan

  # -----------------------
  # Optionally run pipeline_run.py now (keep simple python detection)
  # -----------------------
  if ($RunNow) {
    $pythonExe = $null
    if (Get-Command python -ErrorAction SilentlyContinue) { $pythonExe = "python" }
    elseif (Get-Command python3 -ErrorAction SilentlyContinue) { $pythonExe = "python3" }

    if (-not $pythonExe) {
      Write-Host "Python not found in PATH. Skipping execution of pipeline_run.py. You can run it manually with: python `"$pyPath`"" -ForegroundColor Yellow
    } else {
      Write-Host "Running pipeline_run.py with $pythonExe ..." -ForegroundColor Green
      $proc = Start-Process -FilePath $pythonExe -ArgumentList ("`"$pyPath`"") -WorkingDirectory $HasilDir -NoNewWindow -PassThru -Wait
      if ($proc.ExitCode -ne 0) {
        Write-Host "pipeline_run.py exited with code $($proc.ExitCode)" -ForegroundColor Red
      } else {
        Write-Host "pipeline_run.py finished (exit code 0)" -ForegroundColor Cyan
      }
    }
  } else {
    Write-Host "RunNow = false; files written but pipeline not executed." -ForegroundColor Yellow
    Write-Host "Open the combined report via local HTTP server for best behavior:"
    Write-Host "  cd `"$($HasilDir)/notebooks_html`""
    Write-Host "  python -m http.server 8000"
    Write-Host "Then open http://localhost:8000/report_combined_yoy_and_notebook.html"
    Write-Host "Standalone YoY editor file:"
    Write-Host "  $htmlYoYPath"
  }

} catch {
  Write-Host "Script failed: $_" -ForegroundColor Red
  if ($_.Exception -ne $null) { Write-Host $_.Exception.Message -ForegroundColor Red }
}
```
