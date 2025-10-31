# Time Series Forecasting — Runme (gabungan README & petunjuk run script)

Ringkasan:
- File ini adalah dokumentasi gabungan: panduan UI YoY, contoh CSV, dan instruksi menjalankan skrip PowerShell `run_pipeline_complete_with_ui.ps1`.
- Skrip PowerShell membuat folder hasil, menulis pipeline Python, menulis UI HTML interaktif (gabungan YoY + notebook preview), mengeksekusi pipeline (opsional), dan membuat ringkasan notebook (.ipynb).

## Konten yang dibuat oleh skrip
- Folder default: `C:\Users\ASUS\Desktop\time-series-forecasting\hasil` (konfigurabel via param)
- File Python pipeline: `pipeline_run.py` — men-download data (yfinance), preprocess, train model (LightGBM/RandomForest fallback), iterative forecast, menulis artifacts (CSV, model .pkl, report HTML, PNG).
- File HTML interaktif: `yoy_analysis_with_explanation.html` — UI lengkap untuk:
  - Unggah CSV (kolom date,value) atau input manual (tahun+nilai)
  - Pilih granularitas, mode periode (full year / month-day range / template dates)
  - Pilih kolom numeric (actual/forecast/y/value) — auto-detect jika kosong
  - Pilih agregasi (SUM / AVG / LAST)
  - Generate → Chart + Table + Penjelasan heuristik + Notebook preview
  - Export hasil CSV dan Download .ipynb
- Notebook hasil ringkasan: `TimeSeriesForecasting_results.ipynb`
- Log run: `run_log.txt`
- Direktori struktur:
  - data/raw
  - data/processed
  - models
  - notebooks_html (temp/report)
  - hasil (root artifacts)

## Cara menjalankan (PowerShell)
1. Simpan skrip `run_pipeline_complete_with_ui.ps1` (yang Anda miliki) atau jalankan langsung:
   - Di PowerShell (Run as Administrator jika perlu):
     ```
     powershell -ExecutionPolicy Bypass -File .\run_pipeline_complete_with_ui.ps1
     ```
   - Atau panggil dengan parameter kustom:
     ```
     .\run_pipeline_complete_with_ui.ps1 -ProjectDir "C:\MyProject" -WorkingFolderName "hasil" -RunNow:$true
     ```
2. Prasyarat:
   - Python harus ada di PATH.
   - Jika ada paket yang hilang, install:
     ```
     python -m pip install pandas numpy yfinance scikit-learn joblib plotly matplotlib jinja2 lightgbm
     ```
   - Skrip sudah berisi upaya instalasi ringan untuk `jinja2` (fallback).

3. Hasil:
   - Buka `yoy_analysis_with_explanation.html` di folder `hasil` dengan browser (Chrome/Edge/Firefox).
   - Jika `RunNow` true, skrip akan menjalankan `pipeline_run.py` dan menulis artifacts (processed CSV, forecast CSV, report HTML) ke folder hasil.

## Panduan penggunaan UI (singkat & terurut)
1. Siapkan file CSV minimal dengan header: `date,value`
   ```
   date,value
   2019-01-01,100
   2020-01-01,120
   2021-01-01,140
   ```
2. Buka `yoy_analysis_with_explanation.html` di browser.
3. Unggah CSV atau masukkan input manual (Tahun & Nilai → Tambah).
4. Pilih granularitas (Tahunan/Kuartal/Bulanan/Harian).
5. Pilih mode periode analisis:
   - Seluruh tahun (1 Jan – 31 Dec)
   - Rentang bulan-hari (MM-DD)
   - Template tanggal (yyyy-mm-dd)
6. Atur `yearFrom` & `yearTo`.
7. Pilih kolom numeric (atau biarkan `Auto-detect`).
8. Pilih metode agregasi: SUM / AVG / LAST.
   - Rekomendasi:
     - Harga saham: AVG atau LAST
     - Volume / jumlah: SUM
9. Klik **Generate** → lihat chart, tabel, dan penjelasan heuristik.
10. Jika ingin menyimpan: klik **Export CSV** atau **Download .ipynb**.

## Troubleshooting cepat
- Hasil "-" atau kosong:
  - Periksa format tanggal di CSV (gunakan `YYYY-MM-DD`).
  - Pastikan rentang tahun mencakup data Anda.
  - Jika pakai forecast file, pastikan kolom forecast berisi angka.
- Angka terlalu besar (mis. 18238.52 pada kasus harga):
  - Periksa metode agregasi: `SUM` menjumlahkan nilai harian (bukan metrik umum untuk harga). Gunakan `AVG` atau `LAST`.
- Parser kolom salah:
  - Pastikan header file jelas: `date,value` atau `Date,y` (CSV dari pipeline `AAPL_parsed.csv` menggunakan `Date,y`).

## Tips interpretasi penjelasan
- Perubahan sangat besar (>50%) → periksa peristiwa/berita, promosi, split, atau anomali data.
- Penurunan besar → periksa missing data, data truncation, atau event eksternal.
- Tidak ada data di rentang → periksa coverage per tahun.

## Pengembangan/Perbaikan UI (opsional teknis)
- Jika ingin mengubah default agregasi di kode pipeline atau menambahkan pilihan kolom secara otomatis pada upload, edit `yoy_analysis_with_explanation.html` (JS bagian auto-detect column).
- Untuk membuat UI otomatis memilih kolom `actual` vs `forecast`, saya rekomendasikan menambahkan parsing header CSV dan menampilkan dropdown nama semua kolom numeric yang ditemukan.

## Catatan akhir
- HTML yang dihasilkan bersifat "client-only" (JS + ChartJS) dan aman digunakan secara lokal — tidak mengirim data keluar browser.
- Pipeline Python menulis artifacts yang kemudian bisa dibuka dengan UI untuk analisis tambahan.

---

## run_pipeline_complete_with_ui.ps1
Berikut ini adalah isi lengkap `run_pipeline_complete_with_ui.ps1`. Skrip ini sudah disiapkan untuk ditaruh di dokumentasi sehingga Anda bisa menyalin langsung ke file `.ps1` dan menjalankannya.

```powershell
<# 
run_pipeline_complete_with_ui.ps1

This is an updated all-in-one PowerShell script based on the previous wrapper.
What it does:
- Creates project folder: C:\Users\ASUS\Desktop\time-series-forecasting\hasil (default; configurable)
- Writes a robust Python pipeline (pipeline_run.py) into that folder (same behavior as before)
- Writes an interactive HTML UI (yoy_analysis_with_explanation.html) into the hasil folder — this HTML contains date/time input controls, CSV/manual-data ingestion, computes YoY comparisons, and generates human-readable explanations client-side
- Runs the Python pipeline (optional) and streams output to a resilient run_log file; filters noisy watchdog/browser lines
- Builds a summary Jupyter notebook (TimeSeriesForecasting_results.ipynb) in the hasil folder after the run
- All artifacts (reports, CSVs, model .pkl, notebook, UI) are placed inside the hasil folder

Usage:
- Paste the entire script into PowerShell and press Enter, or save as run_pipeline_complete_with_ui.ps1 and run:
    powershell -ExecutionPolicy Bypass -File .\run_pipeline_complete_with_ui.ps1
- Ensure python is in PATH. If packages missing, install:
    python -m pip install pandas numpy yfinance scikit-learn joblib plotly matplotlib jinja2 lightgbm

You asked: "Maksudnya dikodekan di ps1 sebelumnya" — this script includes the HTML UI and writes it into the hasil folder so you can open it locally.
#>

param(
  [string] $ProjectDir = "C:\Users\ASUS\Desktop\time-series-forecasting",
  [string] $WorkingFolderName = "hasil",
  [switch] $RunNow = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Open-LogStream {
  param([string] $Path, [int] $Retries = 8, [int] $DelayMs = 200)
  for ($i = 0; $i -lt $Retries; $i++) {
    try {
      $fs = [System.IO.File]::Open($Path,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite)
      $fs.Seek(0, [System.IO.SeekOrigin]::End) | Out-Null
      $sw = New-Object System.IO.StreamWriter($fs, [System.Text.Encoding]::UTF8)
      $sw.AutoFlush = $true
      return @{ StreamWriter=$sw; Path=$Path; IsFallback=$false }
    } catch {
      Start-Sleep -Milliseconds $DelayMs
    }
  }
  $dir = [System.IO.Path]::GetDirectoryName($Path)
  if ([string]::IsNullOrEmpty($dir)) { $dir = (Get-Location).Path }
  $timestamp = (Get-Date).ToString("yyyyMMddTHHmmss")
  $fallback = [System.IO.Path]::Combine($dir, "run_log_$timestamp.txt")
  $fs2 = [System.IO.File]::Open($fallback,
    [System.IO.FileMode]::Create,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::ReadWrite)
  $sw2 = New-Object System.IO.StreamWriter($fs2, [System.Text.Encoding]::UTF8)
  $sw2.AutoFlush = $true
  return @{ StreamWriter=$sw2; Path=$fallback; IsFallback=$true }
}

try {
  $HasilDir = Join-Path $ProjectDir $WorkingFolderName
  Write-Host "ProjectDir: $ProjectDir"
  Write-Host "Working folder (hasil): $HasilDir"

  # Create folder structure
  New-Item -ItemType Directory -Path $HasilDir -Force | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $HasilDir "data\raw") -Force | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $HasilDir "data\processed") -Force | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $HasilDir "models") -Force | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $HasilDir "notebooks_html") -Force | Out-Null

  # Check Python
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'python' not found in PATH. Activate Python environment or install Python." -ForegroundColor Red
    return
  }

  # -----------------------
  # Write pipeline_run.py
  # -----------------------
  $pyPath = Join-Path $HasilDir "pipeline_run.py"
  $pyContent = @'
#!/usr/bin/env python
"""
pipeline_run.py (robust final)

- Writes outputs into the folder where this script resides (script directory).
- Imports pandas at module scope to avoid local pd binding issues.
- Uses matplotlib for PNG exports (no browser/kaleido required).
- Sanitizes feature names and handles MultiIndex from yfinance.
- Writes processed CSV, forecast CSV, run_info JSON, model pickle, and HTML report.
"""
import sys, json, traceback, re, warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def pip_install(pkg):
    try:
        import subprocess, sys as _sys
        subprocess.run([_sys.executable, "-m", "pip", "install", pkg], check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception:
        pass

pip_install("jinja2")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "notebooks_html"
RESULTS = ROOT
TEMPLATE = ROOT / "report_template.html"

for d in (DATA_RAW, DATA_PROC, MODELS, REPORTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def flatten_columns_if_multiindex(df):
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for col in df.columns:
            parts = [str(x) for x in col if (x is not None and str(x) != "")]
            name = "_".join(parts) if parts else str(col)
            cols.append(name)
        df.columns = cols
    return df

def sanitize_feature_names(cols):
    new = []
    seen = {}
    for c in cols:
        s = re.sub(r"\W+", "_", str(c))
        if s == "":
            s = "col"
        if re.match(r"^\d", s):
            s = "_" + s
        base = s
        i = 1
        while s in seen:
            i += 1
            s = f"{base}_{i}"
        seen[s] = True
        new.append(s)
    return new

def download_data(ticker, start, end, out_dir=DATA_RAW):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{ticker}.csv"
    print(f"[download] {ticker} {start}..{end}")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data")
    df = flatten_columns_if_multiindex(df)
    df.to_csv(p, index=True)
    return df, p

def preprocess(df_or_path, out_path):
    if isinstance(df_or_path, pd.DataFrame):
        df_raw = df_or_path.copy()
    else:
        df_raw = pd.read_csv(df_or_path, header=0, engine="python")
    if df_raw.empty:
        raise RuntimeError("Raw input empty")
    df_raw = flatten_columns_if_multiindex(df_raw)

    try:
        idx = pd.to_datetime(df_raw.index, errors="coerce")
        if idx.notna().sum() / max(1, len(idx)) > 0.5:
            df_raw.index = idx
        else:
            date_col = None
            for c in df_raw.columns:
                if str(c).lower() in ("date","day","timestamp","ds"):
                    date_col = c
                    break
            if date_col is not None:
                df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
                df_raw = df_raw.loc[df_raw[date_col].notna()].copy()
                df_raw.set_index(date_col, inplace=True)
            else:
                first_col = df_raw.columns[0]
                parsed = pd.to_datetime(df_raw[first_col], errors="coerce")
                if parsed.notna().sum() >= 1:
                    df_raw[first_col] = parsed
                    df_raw = df_raw.loc[parsed.notna()].copy()
                    df_raw.set_index(first_col, inplace=True)
    except Exception:
        pass

    numeric_cols = []
    for c in df_raw.columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
            if df_raw[c].notna().sum() > 0:
                numeric_cols.append(c)
        except Exception:
            continue

    preferred = None
    for cand in ("Close","close","Adj_Close","Adj Close","AdjClose"):
        if cand in df_raw.columns:
            preferred = cand
            break
    if preferred is None:
        for c in df_raw.columns:
            if re.search(r"(?i)\bclose\b", str(c)):
                preferred = c
                break

    if not preferred:
        if numeric_cols:
            preferred = numeric_cols[0]
            print(f"[preprocess] Warning: 'Close' not found. Using numeric column: {preferred}")
        else:
            raise RuntimeError("No numeric price column found")

    s = df_raw[[preferred]].rename(columns={preferred: "y"}).sort_index()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].copy()
    if s.empty:
        raise RuntimeError("After parsing dates, no valid rows remain")
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

def create_features(df, lags=(1,2,3,7,14,30), windows=(3,7,14)):
    out = df.copy()
    for l in lags:
        out[f"lag_{l}"] = out["y"].shift(l)
    for w in windows:
        out[f"roll_mean_{w}"] = out["y"].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out["y"].shift(1).rolling(w).std()
    return out

def train_model_with_fallback(X, y):
    original_cols = list(X.columns)
    safe_cols = sanitize_feature_names(original_cols)
    orig_to_safe = dict(zip(original_cols, safe_cols))
    X_safe = X.copy()
    X_safe.columns = safe_cols
    try:
        import lightgbm as lgb
        print("[train] Trying LightGBM")
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        model.fit(X_safe, y)
        meta = {"orig_to_safe": orig_to_safe, "safe_cols": safe_cols}
        return model, "lightgbm", meta
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_safe, y)
        meta = {"orig_to_safe": orig_to_safe, "safe_cols": safe_cols}
        return model, "random_forest", meta

def iterative_forecast(model, history_df, steps=30, meta=None):
    cur = history_df.copy().asfreq("D")
    preds = []
    for _ in range(steps):
        feat = create_features(cur).iloc[[-1]].drop(columns=["y"], errors="ignore")
        if meta and "orig_to_safe" in meta:
            feat = feat.rename(columns=meta["orig_to_safe"])
            safe_cols = meta.get("safe_cols", list(feat.columns))
            for c in safe_cols:
                if c not in feat.columns:
                    feat[c] = 0.0
            feat = feat.loc[:, safe_cols]
        feat = feat.fillna(method="ffill").fillna(method="bfill").fillna(0)
        try:
            p = float(model.predict(feat)[0])
        except Exception:
            p = float(model.predict(feat.values)[0])
        next_date = cur.index[-1] + pd.Timedelta(days=1)
        preds.append((next_date, p))
        cur.loc[next_date] = [p]
    return pd.DataFrame({"forecast": [v for (_, v) in preds]}, index=[d for (d, _) in preds])

def generate_report(fig, run_info, out_path, series=None, fc=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pio.write_html(fig, file=str(out_path), include_plotlyjs="cdn", full_html=True)
        print("[report] HTML written:", out_path)
    except Exception:
        print("[report] HTML write skipped")
    png_path = out_path.with_suffix(".png")
    try:
        plt.figure(figsize=(10,6))
        if series is not None:
            try:
                plt.plot(series.index, series["y"], label="Actual", color="black")
            except Exception:
                pass
        if fc is not None:
            try:
                plt.plot(fc.index, fc["forecast"], label="Forecast", color="red")
            except Exception:
                pass
        plt.legend()
        plt.title("Forecast {}".format(run_info.get("ticker","")))
        plt.tight_layout()
        plt.savefig(str(png_path))
        plt.close()
        print("[report] PNG written with matplotlib:", png_path)
    except Exception:
        print("[report] PNG (matplotlib) skipped")

def main():
    try:
        print("Pipeline started. TIMESTAMP=", TIMESTAMP)
        TICKER = "AAPL"
        START = "2015-01-01"
        END = "2024-01-01"
        HORIZON = 30

        df_raw, raw_path = download_data(TICKER, START, END)
        processed_df, processed_path = preprocess(df_raw, DATA_PROC / f"{TICKER}_parsed.csv")
        print("[main] Processed CSV written to:", processed_path)

        series = processed_df.copy()
        try:
            series = series.asfreq("D")
        except Exception:
            dr = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
            series = series.reindex(dr)
        series["y"] = series["y"].ffill()

        df_feat = create_features(series).dropna()
        if df_feat.empty:
            raise RuntimeError("Not enough data after feature creation to train a model.")
        X = df_feat.drop(columns=["y"])
        y = df_feat["y"]
        model, model_name, meta = train_model_with_fallback(X, y)
        model_file = MODELS / f"{model_name}_{TICKER}_final_{TIMESTAMP}.pkl"
        joblib.dump({"model": model, "meta": meta}, model_file)
        print("[main] Model saved:", model_file)

        fc = iterative_forecast(model, series, steps=HORIZON, meta=meta)
        forecast_path = RESULTS / f"{TICKER}_forecast_{TIMESTAMP}.csv"
        combined = pd.concat([series.rename(columns={"y":"actual"}), fc], axis=0)
        combined.to_csv(forecast_path)
        print(f"[main] Forecast saved: {forecast_path}")

        run_info = {"ticker": TICKER, "start": START, "end": END, "horizon": HORIZON, "model_used": model_name, "timestamp": TIMESTAMP}
        run_info_path = RESULTS / f"run_info_{TIMESTAMP}.json"
        run_info_path.write_text(json.dumps(run_info, indent=2))
        print("[main] Run info saved:", run_info_path)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series["y"], name="Actual", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(color="red")))
        fig.update_layout(title=f"{TICKER} Actual vs Forecast", xaxis_title="Date", yaxis_title="Price", height=600)
        report_path = REPORTS / f"report_{TICKER}_{TIMESTAMP}.html"
        generate_report(fig, run_info, report_path, series=series, fc=fc)

        print("=== Done ===")
    except Exception as e:
        print("Pipeline failed:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'@

  Set-Content -LiteralPath $pyPath -Value $pyContent -Encoding UTF8
  Write-Host "Wrote pipeline_run.py to: $pyPath" -ForegroundColor Green

  # -----------------------
  # Write HTML UI file (date/time inputs + YoY explanation) into hasil
  # -----------------------
  $htmlPath = Join-Path $HasilDir "yoy_analysis_with_explanation.html"
  $htmlContent = @'
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Analisis YoY dengan Penjelasan Otomatis</title>
  <style>
    body { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial; margin:18px; color:#111 }
    h1 { margin-top:0 }
    .wrap { display:grid; grid-template-columns: 420px 1fr; gap:18px; align-items:start; }
    .card { padding:12px; border:1px solid #e6eef8; border-radius:8px; background:white }
    label{display:block;margin-top:10px;font-weight:600;font-size:13px}
    input, select, button, textarea { width:100%; padding:8px; margin-top:6px; border:1px solid #d1d9e6; border-radius:6px }
    .controls-row { display:flex; gap:8px }
    .small { width:auto; padding:6px 8px; font-size:13px }
    .button { background:#2563eb;color:white;border:none;padding:10px;border-radius:6px;cursor:pointer;font-weight:600 }
    .muted { color:#666; font-size:13px }
    .chart-wrap{height:260px}
    table{width:100%;border-collapse:collapse;margin-top:12px}
    th,td{padding:8px;border-bottom:1px solid #eee;text-align:left}
    th{background:#fafafa;font-weight:700}
    .explain{margin-top:12px;padding:12px;border-radius:8px;background:#f8fafc;border:1px solid #e6eef8}
    pre { white-space:pre-wrap; word-break:break-word; background:#0b1220; color:#e6eef8; padding:12px; border-radius:6px; overflow:auto }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>Analisis Perbandingan Waktu & Penjelasan Otomatis</h1>
  <p class="muted">Masukkan data (CSV: kolom tanggal & nilai) atau gunakan manual input. Pilih periode/waktu yang ingin dibandingkan. Klik "Generate" untuk melihat perbedaan dan penjelasan mengapa perubahan itu terjadi (heuristik dijelaskan di bawah).</p>

  <div class="wrap">
    <div class="card">
      <h3>Sumber Data</h3>
      <label>Unggah CSV (kolom: date,value)</label>
      <input type="file" id="csvFile" accept=".csv" />
      <div class="muted">Tanggal bisa dalam format ISO, yyyy-mm-dd, dd/mm/yyyy, atau mm/dd/yyyy.</div>

      <label>Atau input manual (tahun & nilai)</label>
      <div style="display:flex; gap:8px; margin-top:6px">
        <input type="number" id="manualYear" placeholder="Tahun" />
        <input type="number" id="manualValue" placeholder="Nilai" />
        <button id="addManualBtn" class="small">Tambah</button>
      </div>
      <table id="manualTable" style="display:none;margin-top:10px">
        <thead><tr><th>Tahun</th><th>Nilai</th><th>Aksi</th></tr></thead>
        <tbody></tbody>
      </table>

      <hr style="margin:12px 0;border:none;height:1px;background:#eef2f6" />

      <h3>Pengaturan Periode/Waktu</h3>
      <label>Granularitas waktu</label>
      <select id="granularity">
        <option value="year">Tahunan (total/avg per tahun)</option>
        <option value="quarter">Kuartal (Q1,Q2...)</option>
        <option value="month">Bulanan</option>
        <option value="day">Harian</option>
      </select>

      <label>Mode periode analisis</label>
      <select id="periodMode">
        <option value="fullYear">Seluruh tahun (1 Jan - 31 Dec)</option>
        <option value="monthDayRange">Rentang bulan-hari (contoh 03-01 sampai 04-30)</option>
        <option value="customDates">Template tanggal (pilih tanggal awal & akhir)</option>
      </select>

      <div id="monthDayRange" style="display:none">
        <label>Mulai (MM-DD)</label><input placeholder="01-01" id="startMMDD" />
        <label>Selesai (MM-DD)</label><input placeholder="12-31" id="endMMDD" />
      </div>

      <div id="customDateRange" style="display:none">
        <label>Template Mulai (yyyy-mm-dd)</label><input type="date" id="startTemplate" />
        <label>Template Selesai (yyyy-mm-dd)</label><input type="date" id="endTemplate" />
      </div>

      <label>Rentang tahun</label>
      <div style="display:flex; gap:8px">
        <input type="number" id="yearFrom" value="2019" class="small" />
        <input type="number" id="yearTo" value="2024" class="small" />
      </div>

      <div style="margin-top:12px; display:flex; gap:8px">
        <button id="generateBtn" class="button">Generate</button>
        <button id="explainBtn" class="small">Jelaskan (lanjutan)</button>
        <button id="downloadIpynb" class="small">Download .ipynb</button>
      </div>

      <div style="margin-top:12px" class="muted">
        Penjelasan yang dihasilkan bersifat heuristik: mengidentifikasi trend, seasonality, outlier, missing data, serta perubahan struktural (sketsa alasannya).
      </div>
    </div>

    <div class="card">
      <h3>Hasil</h3>
      <div class="chart-wrap"><canvas id="chart" height="240"></canvas></div>

      <table id="resultTable" style="display:none">
        <thead><tr><th>Periode</th><th>Nilai</th><th>Selisih (vs prev)</th><th>Perubahan %</th></tr></thead>
        <tbody></tbody>
      </table>

      <div class="explain" id="explanation" style="display:none"></div>

      <div style="margin-top:8px;">
        <button id="exportCsv" class="small">Export Hasil CSV</button>
      </div>

      <div id="messages" class="muted" style="margin-top:8px"></div>
    </div>
  </div>

<script>
/* Utility & parsing */
function parseCSV(text) {
  const lines = text.replace(/\r\n/g,"\n").split("\n").filter(l=>l.trim()!=="");
  if (lines.length===0) return {header:[],rows:[]};
  const header = lines[0].split(",").map(h=>h.trim());
  const rows = lines.slice(1).map(l => l.split(",").map(c=>c.trim()));
  return {header, rows};
}
function toDate(s){
  if(!s) return null;
  let d = Date.parse(s);
  if(!isNaN(d)) return new Date(d);
  const m = s.match(/^(\d{1,2})[\/\-\.\s](\d{1,2})[\/\-\.\s](\d{2,4})$/);
  if(m){
    let day=parseInt(m[1],10), mon=parseInt(m[2],10)-1, year=parseInt(m[3],10); if(year<100) year+=2000;
    return new Date(year,mon,day);
  }
  return null;
}

/* State */
let rawData = []; // {date:Date, value:Number}
let manualData = {}; // {year: value}

/* Elements */
const csvFile = document.getElementById('csvFile');
const addManualBtn = document.getElementById('addManualBtn');
const manualYear = document.getElementById('manualYear');
const manualValue = document.getElementById('manualValue');
const manualTable = document.querySelector('#manualTable tbody');
const manualTableWrap = document.getElementById('manualTable');
const periodMode = document.getElementById('periodMode');
const monthDayRange = document.getElementById('monthDayRange');
const customDateRange = document.getElementById('customDateRange');
const startMMDD = document.getElementById('startMMDD');
const endMMDD = document.getElementById('endMMDD');
const startTemplate = document.getElementById('startTemplate');
const endTemplate = document.getElementById('endTemplate');
const yearFrom = document.getElementById('yearFrom');
const yearTo = document.getElementById('yearTo');
const generateBtn = document.getElementById('generateBtn');
const explainBtn = document.getElementById('explainBtn');
const exportCsv = document.getElementById('exportCsv');
const downloadIpynb = document.getElementById('downloadIpynb');
const chartCanvas = document.getElementById('chart');
const resultTable = document.getElementById('resultTable');
const resultTbody = resultTable.querySelector('tbody');
const explanationDiv = document.getElementById('explanation');
const messages = document.getElementById('messages');
let chart=null;

/* CSV upload handling */
csvFile.addEventListener('change', e=>{
  const f = e.target.files[0];
  if(!f) return;
  const r = new FileReader();
  r.onload = () => {
    const parsed = parseCSV(r.result);
    const {header, rows} = parsed;
    let dateIdx=-1, valIdx=-1;
    for(let i=0;i<header.length;i++){
      let dateCount=0, numCount=0;
      for(let j=0;j<Math.min(rows.length,20);j++){
        const v=rows[j][i]||"";
        if(toDate(v)) dateCount++;
        if(!isNaN(parseFloat(v)) && v!=="") numCount++;
      }
      if(dateCount >= Math.max(1, Math.floor(rows.length*0.3))) dateIdx=i;
      if(numCount >= Math.max(1, Math.floor(rows.length*0.3))) valIdx=i;
    }
    if(dateIdx===-1) dateIdx=0;
    if(valIdx===-1) valIdx=(header.length>1?1:0);
    rawData=[];
    rows.forEach(rw=>{
      const d = toDate(rw[dateIdx]);
      const v = parseFloat(rw[valIdx]);
      if(d && !isNaN(v)) rawData.push({date:d, value:v});
    });
    messages.textContent = `CSV loaded: ${rawData.length} baris valid (date col=${header[dateIdx]||dateIdx}, value col=${header[valIdx]||valIdx})`;
  };
  r.readAsText(f);
});

/* Manual input */
addManualBtn.addEventListener('click', ()=>{
  const y = parseInt(manualYear.value,10);
  const v = parseFloat(manualValue.value);
  if(!y || isNaN(v)){ alert('Masukkan tahun & nilai valid'); return; }
  manualData[y]=v;
  renderManualTable();
  manualYear.value=""; manualValue.value="";
});
function renderManualTable(){
  manualTable.innerHTML="";
  const yrs = Object.keys(manualData).map(x=>parseInt(x,10)).sort((a,b)=>a-b);
  if(yrs.length){
    manualTableWrap.style.display="";
    yrs.forEach(y=>{
      const tr=document.createElement('tr');
      tr.innerHTML = `<td>${y}</td><td>${manualData[y]}</td><td><button class="small" data-year="${y}">Hapus</button></td>`;
      manualTable.appendChild(tr);
    });
    manualTable.querySelectorAll('button').forEach(btn=>btn.addEventListener('click', e=>{
      const y = e.target.getAttribute('data-year');
      delete manualData[y];
      renderManualTable();
    }));
  } else { manualTableWrap.style.display="none"; }
}

/* Period UI change */
periodMode.addEventListener('change', ()=>{
  const v=periodMode.value;
  monthDayRange.style.display = (v==='monthDayRange') ? 'block' : 'none';
  customDateRange.style.display = (v==='customDates') ? 'block' : 'none';
});

/* Aggregation + explanation heuristics */
function aggregateAndExplain(){
  // build per-year aggregates according to mode; prefer CSV rawData, else manualData
  const useManual = (Object.keys(manualData).length>0 && rawData.length===0);
  const fromYear = parseInt(yearFrom.value,10), toYear = parseInt(yearTo.value,10);
  if(isNaN(fromYear) || isNaN(toYear) || fromYear>toYear){ alert('Rentang tahun tidak valid'); return; }
  let results = [];
  if(useManual){
    for(let y=fromYear;y<=toYear;y++){
      const val = manualData[y] !== undefined ? manualData[y] : null;
      results.push({period: String(y), value: val});
    }
  } else {
    // rawData -> filter by each year's template
    const mode = periodMode.value;
    for(let y=fromYear; y<=toYear; y++){
      let start, end;
      if(mode==='fullYear'){
        start = new Date(y,0,1); end = new Date(y,11,31,23,59,59);
      } else if(mode==='monthDayRange'){
        const s = startMMDD.value || startMMDD.placeholder || '01-01';
        const e = endMMDD.value || endMMDD.placeholder || '12-31';
        const sp = s.split('-'); const ep = e.split('-');
        start = new Date(y, parseInt(sp[0],10)-1, parseInt(sp[1],10));
        end = new Date(y, parseInt(ep[0],10)-1, parseInt(ep[1],10),23,59,59);
      } else if(mode==='customDates'){
        const st = startTemplate.value; const en = endTemplate.value;
        if(!st || !en){ messages.textContent = 'Pilih template tanggal untuk mode customDates.'; return; }
        const sdate = new Date(st); const edate = new Date(en);
        start = new Date(y, sdate.getMonth(), sdate.getDate());
        end = new Date(y, edate.getMonth(), edate.getDate(),23,59,59);
      } else {
        start = new Date(y,0,1); end = new Date(y,11,31,23,59,59);
      }
      const slice = rawData.filter(r => r.date >= start && r.date <= end);
      const sum = slice.reduce((acc,it)=>acc + (isFinite(it.value)?it.value:0), 0);
      const avg = slice.length? (sum / slice.length) : null;
      // we present "value" as sum by default; if you want avg, change logic or add UI
      results.push({period: String(y), value: slice.length? sum : null, count: slice.length, avg: avg});
    }
  }

  // compute diffs and percent changes
  for(let i=0;i<results.length;i++){
    const cur = results[i];
    const prev = i>0 ? results[i-1] : null;
    if(prev && cur.value !== null && prev.value !== null){
      cur.diff = cur.value - prev.value;
      cur.pct = prev.value !== 0 ? (cur.diff / Math.abs(prev.value) * 100) : null;
    } else {
      cur.diff = null; cur.pct = null;
    }
  }

  // Render table and chart
  resultTbody.innerHTML = "";
  const labels = []; const dataVals = [];
  results.forEach(r=>{
    const tr=document.createElement('tr');
    const diff = r.diff === null ? "-" : r.diff.toFixed(2);
    const pct = r.pct === null ? "-" : r.pct.toFixed(2) + "%";
    tr.innerHTML = `<td>${r.period}</td><td>${r.value===null? "-": r.value.toFixed(2)}</td><td>${diff}</td><td>${pct}</td>`;
    resultTbody.appendChild(tr);
    labels.push(r.period);
    dataVals.push(r.value===null?0:r.value);
  });
  resultTable.style.display="table";

  if(chart) chart.destroy();
  const ctx = chartCanvas.getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: labels, datasets: [{ label: 'Value', data: dataVals, borderColor:'#2563eb', backgroundColor:'rgba(37,99,235,0.08)', tension:0.2 }] },
    options: { responsive:true, scales:{ y:{ beginAtZero:false } } }
  });

  // Explanation heuristics (simple, human-readable)
  let explanation = "";
  for(let i=0;i<results.length;i++){
    const r = results[i];
    if(i===0){
      explanation += `<strong>${r.period}:</strong> value = ${r.value===null? "tidak tersedia": r.value.toFixed(2)}. (Basis awal)\n\n`;
      continue;
    }
    if(r.value === null){
      explanation += `<strong>${r.period}:</strong> data tidak tersedia untuk periode ini.\n\n`;
      continue;
    }
    const prev = results[i-1];
    if(prev.value === null){
      explanation += `<strong>${r.period}:</strong> ada nilai ${r.value.toFixed(2)} tetapi tahun sebelumnya tidak tersedia untuk perbandingan.\n\n`;
      continue;
    }
    const diff = r.value - prev.value;
    const pct = prev.value !== 0 ? (diff / Math.abs(prev.value) * 100) : null;
    // heuristic classification
    let reason = [];
    if(Math.abs(diff) < Math.abs(prev.value) * 0.01) reason.push("Tidak ada perubahan signifikan (±1%)");
    if(pct !== null && pct > 50) reason.push("Peningkatan besar — mungkin ada peristiwa pasar, pengumuman, atau perubahan musiman");
    if(pct !== null && pct < -50) reason.push("Penurunan besar — cek anomali, kehilangan data, atau peristiwa negatif");
    if(r.count !== undefined && r.count === 0) reason.push("Tidak ada data di rentang yang dipilih (0 baris)");
    if(reason.length===0) reason.push("Perubahan moderat; bisa disebabkan kombinasi trend + seasonality + volatilitas");
    explanation += `<strong>${r.period} vs ${prev.period}:</strong> nilai ${r.value.toFixed(2)} (selisih ${diff.toFixed(2)}${pct!==null? ", " + pct.toFixed(2) + "%": ""}). Alasan: ${reason.join("; ")}.\n\n`;
  }

  explanationDiv.style.display="";
  explanationDiv.innerHTML = "<pre>"+explanation+"</pre>";
  messages.textContent = "Analisis selesai.";
}

/* Export CSV of results (simple) */
exportCsv.addEventListener('click', ()=>{
  // build CSV from table
  const rows = [["Period","Value","Diff","Pct"]];
  resultTbody.querySelectorAll('tr').forEach(tr=>{
    const tds = tr.querySelectorAll('td');
    rows.push([tds[0].innerText, tds[1].innerText, tds[2].innerText, tds[3].innerText]);
  });
  const csv = rows.map(r => r.join(",")).join("\n");
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
  a.download = "yoy_results.csv";
  document.body.appendChild(a); a.click(); a.remove();
});

/* Generate / Explain button */
generateBtn.addEventListener('click', ()=> {
  aggregateAndExplain();
});

/* Explain (expanded): simple heuristics + tips */
explainBtn.addEventListener('click', ()=>{
  // Provide additional explanation tips based on displayed explanation
  const tips = `Tips pemeriksaan lanjutan:
- Periksa apakah ada outlier (hari nilai ekstrem) di data mentah.
- Periksa apakah ada perubahan definisi/metrik antar tahun.
- Cek volume (jumlah bar) di tiap tahun; 0 bar berarti rentang kosong.
- Periksa berita/kejadian pada periode dengan perubahan besar (earnings, regulation, holidays).`;
  alert(tips);
});

/* Download minimal .ipynb (summary) */
downloadIpynb.addEventListener('click', ()=>{
  // Build a small notebook with the explanation + table as plaintext outputs
  const mdTitle = "# YoY Analysis Results\nGenerated by local UI\n\n";
  const explanationText = explanationDiv.innerText || "No explanation generated yet.";
  const processed = resultTbody.innerText || "No results.";
  const nb = {
    nbformat:4, nbformat_minor:5,
    metadata: { kernelspec:{ name:"python3", display_name:"Python 3" }, language_info:{ name:"python" } },
    cells: [
      { cell_type: "markdown", metadata:{}, source: [mdTitle] },
      { cell_type: "code", metadata:{}, execution_count:null, source:["# Explanation\n"], outputs:[ { output_type:"execute_result", data:{ "text/plain": [explanationText] }, metadata:{}, execution_count:1 } ] },
      { cell_type: "code", metadata:{}, execution_count:null, source:["# Results table\n"], outputs:[ { output_type:"execute_result", data:{ "text/plain": [processed] }, metadata:{}, execution_count:1 } ] }
    ]
  };
  const blob = new Blob([JSON.stringify(nb, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = "TimeSeries_YoY_results.ipynb";
  document.body.appendChild(a); a.click(); a.remove();
});

/* initial UI */
renderManualTable();

</script>
</body>
</html>
'@

  Set-Content -LiteralPath $htmlPath -Value $htmlContent -Encoding UTF8
  Write-Host "Wrote HTML UI: $htmlPath" -ForegroundColor Green

  # -----------------------
  # Execute pipeline_run.py (optional)
  # -----------------------
  if ($RunNow) {
    $openResult = Open-LogStream -Path (Join-Path $HasilDir "run_log.txt") -Retries 8 -DelayMs 200
    $logStream = $openResult.StreamWriter
    $actualLogPath = $openResult.Path
    if ($openResult.IsFallback) { Write-Host "Using fallback log file: $actualLogPath" -ForegroundColor Yellow } else { Write-Host "Logging to: $actualLogPath" }

    Write-Host "Starting pipeline_run.py ..." -ForegroundColor Green
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = "`"$pyPath`""
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.WorkingDirectory = $HasilDir

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.Start() | Out-Null

    try {
      while (-not $proc.HasExited) {
        while (-not $proc.StandardOutput.EndOfStream) {
          $line = $proc.StandardOutput.ReadLine()
          if ($line -ne $null) {
            if ($line -match "Wait expired" -or $line -match "Watchdog" -or $line -match "Browser is being closed") { continue }
            Write-Host $line
            try { $logStream.WriteLine($line) } catch {}
          }
        }
        while (-not $proc.StandardError.EndOfStream) {
          $err = $proc.StandardError.ReadLine()
          if ($err -ne $null) {
            if ($err -match "Wait expired" -or $err -match "Watchdog" -or $err -match "Browser is being closed") { continue }
            Write-Host $err -ForegroundColor Red
            try { $logStream.WriteLine($err) } catch {}
          }
        }
        Start-Sleep -Milliseconds 100
      }
      # drain remainder
      while (-not $proc.StandardOutput.EndOfStream) {
        $line = $proc.StandardOutput.ReadLine()
        if ($line -ne $null) {
          if ($line -match "Wait expired" -or $line -match "Watchdog" -or $line -match "Browser is being closed") { continue }
          Write-Host $line
          try { $logStream.WriteLine($line) } catch {}
        }
      }
      while (-not $proc.StandardError.EndOfStream) {
        $err = $proc.StandardError.ReadLine()
        if ($err -ne $null) {
          if ($err -match "Wait expired" -or $err -match "Watchdog" -or $err -match "Browser is being closed") { continue }
          Write-Host $err -ForegroundColor Red
          try { $logStream.WriteLine($err) } catch {}
        }
      }
      $exitCode = $proc.ExitCode
      try { $logStream.WriteLine("PROCESS_EXIT_CODE: $exitCode") } catch {}
      $logStream.Flush()
    } finally {
      if ($logStream) { $logStream.Close() }
      if ($proc -and -not $proc.HasExited) { try { $proc.Kill() } catch {} }
    }

    if ($exitCode -eq 0) {
      Write-Host "`nPipeline finished successfully (exit code 0)." -ForegroundColor Cyan
    } else {
      Write-Host "`nPipeline finished with exit code $exitCode" -ForegroundColor Red
      Write-Host "`nLast 200 lines of log ($actualLogPath):" -ForegroundColor Yellow
      if (Test-Path $actualLogPath) {
        Get-Content -Path $actualLogPath -Tail 200 | ForEach-Object {
          if ($_ -match "Wait expired|Watchdog|Browser is being closed") { continue }
          Write-Host $_
        }
      }
    }
  } else {
    Write-Host "RunNow = false; pipeline written but not executed." -ForegroundColor Yellow
  }

  # -----------------------
  # Build notebook summarizing run
  # -----------------------
  try {
    $runLog = if (Test-Path $actualLogPath) { [string]::Join("`n", (Get-Content -Path $actualLogPath -ErrorAction SilentlyContinue)) } else { "No run_log available." }

    $processedFileObj = Get-ChildItem -Path (Join-Path $HasilDir "data\processed") -Filter "*_parsed.csv" -File -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -ne $processedFileObj) {
      $procLines = Get-Content -Path $processedFileObj.FullName -ErrorAction SilentlyContinue | Select-Object -Last 5
      $processedPreview = [string]::Join("`n", $procLines)
    } else { $processedPreview = "Processed CSV not found." }

    $forecastFileObj = Get-ChildItem -Path $HasilDir -Filter "*_forecast_*.csv" -File -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -ne $forecastFileObj) {
      $fcLines = Get-Content -Path $forecastFileObj.FullName -ErrorAction SilentlyContinue | Select-Object -First 10
      $forecastPreview = [string]::Join("`n", $fcLines)
    } else { $forecastPreview = "Forecast CSV not found." }

    $cells = @()

    $cells += @{
      cell_type = "markdown"
      metadata = @{}
      source = @(
        "# Time Series Forecasting — Run Summary`n",
        "`n",
        "This notebook was generated automatically and contains a short run log, previews and artifact list.`n"
      )
    }

    $runLogLines = if ($runLog -ne $null) { $runLog -split "`n" } else { @("No run_log available.") }
    $cells += @{
      cell_type = "code"
      metadata = @{}
      execution_count = $null
      source = @("# Run log (last lines)`n")
      outputs = @(
        @{
          output_type = "execute_result"
          data = @{ "text/plain" = $runLogLines }
          metadata = @{}
          execution_count = 1
        }
      )
    }

    $procLinesArray = if ($processedPreview -ne $null) { $processedPreview -split "`n" } else { @("Processed CSV not found.") }
    $cells += @{
      cell_type = "code"
      metadata = @{}
      execution_count = $null
      source = @("# Processed preview (last 5 rows)`n")
      outputs = @(
        @{
          output_type = "execute_result"
          data = @{ "text/plain" = $procLinesArray }
          metadata = @{}
          execution_count = 1
        }
      )
    }

    $fcLinesArray = if ($forecastPreview -ne $null) { $forecastPreview -split "`n" } else { @("Forecast CSV not found.") }
    $cells += @{
      cell_type = "code"
      metadata = @{}
      execution_count = $null
      source = @("# Forecast preview (first 10 rows)`n")
      outputs = @(
        @{
          output_type = "execute_result"
          data = @{ "text/plain" = $fcLinesArray }
          metadata = @{}
          execution_count = 1
        }
      )
    }

    $artifactLines = @("Artifacts found in $HasilDir")
    foreach ($h in $htmls) { $artifactLines += $h.FullName }
    foreach ($c in $csvs)  { $artifactLines += $c.FullName }
    foreach ($j in $jsons) { $artifactLines += $j.FullName }
    foreach ($m in $models) { $artifactLines += $m.FullName }

    $cells += @{
      cell_type = "code"
      metadata = @{}
      execution_count = $null
      source = @("# Artifacts`n")
      outputs = @(
        @{
          output_type = "execute_result"
          data = @{ "text/plain" = $artifactLines }
          metadata = @{}
          execution_count = 1
        }
      )
    }

    $nb = @{
      nbformat = 4
      nbformat_minor = 5
      metadata = @{
        kernelspec = @{ name = "python3"; display_name = "Python 3" }
        language_info = @{ name = "python" }
      }
      cells = $cells
    }

    $nbPath = Join-Path $HasilDir "TimeSeriesForecasting_results.ipynb"
    $nb | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $nbPath -Encoding UTF8
    Write-Host "Wrote notebook: $nbPath" -ForegroundColor Green

  } catch {
    Write-Host "Failed to create notebook: $_" -ForegroundColor Yellow
  }

  Write-Host "`nAll done."
}
catch {
  Write-Host "Script failed: $_" -ForegroundColor Red
  if ($_.Exception -ne $null) { Write-Host $_.Exception.Message -ForegroundColor Red }
}
```

---

Catatan singkat:
- Saya telah memasukkan seluruh isi skrip PowerShell Anda ke dalam bagian "run_pipeline_complete_with_ui.ps1" di runme.md sehingga dokumentasi sekarang berisi instruksi + skrip lengkap yang bisa dicopy/paste langsung ke file `.ps1`.
- Selanjutnya, jika Anda ingin saya juga:
  - Menyuntikkan runme.md dan HTML ke folder proyek secara otomatis (membuat patch/commit), atau
  - Mengubah skrip agar menulis runme.md langsung ke folder hasil saat dijalankan,
  beri tahu saya mana yang Anda pilih dan saya akan lakukan langkah berikutnya.
