# Time-Series-Forecasting

Pipeline end-to-end untuk meramal deret waktu (saham, penjualan, cuaca).  
Mencakup: pengunduhan data, pra-proses, feature engineering, model (ARIMA / Prophet / LightGBM / LSTM), backtesting rolling-origin, evaluasi, dan visualisasi. README ini juga menyertakan skrip PowerShell satu-blok (salin-tempel) untuk inisialisasi Git & push ke GitHub tanpa membuat file .ps1.

Author: Roberto Ocaviantyo Tahta Laksmana  
Email: yirassssindaba@gmail.com  
Repo target (contoh): https://github.com/yirassssindaba-coder/Time-Series-Forecasting

---

## Struktur proyek (disarankan)
- README.md (ini)
- requirements.txt
- .gitignore
- LICENSE
- data/
  - raw/
  - processed/
- notebooks/
  - 01_EDA.ipynb
  - 02_modeling.ipynb
  - report.ipynb
- src/
  - data/download_data.py
  - preprocess.py
  - features/feature_engineering.py
  - models/
    - lgbm_model.py
    - prophet_model.py
    - lstm_model.py
  - evaluation/backtest.py
  - viz/
- scripts/
  - run_demo.sh
  - (opsional) generate_results.ps1

---

## Prasyarat
- Python 3.9–3.11
- Git (pastikan `git --version` berhasil)
- Jika menggunakan Prophet di Windows, disarankan pakai Conda (conda-forge)
- Jika ingin LSTM pada GPU, pastikan CUDA/cuDNN sesuai dengan versi TensorFlow

Dependensi inti (tambahkan ke `requirements.txt`):
- pandas, numpy, scikit-learn, lightgbm, prophet, tensorflow, statsmodels, yfinance, matplotlib, seaborn, jupyterlab

---

## Instalasi (cepat)
1. Clone repo:
```bash
git clone https://github.com/yirassssindaba-coder/Time-Series-Forecasting.git
cd Time-Series-Forecasting
```

2. Virtualenv (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. (Alternatif Conda untuk Prophet/TensorFlow GPU):
```bash
conda create -n tsforecast python=3.10 -y
conda activate tsforecast
conda install -c conda-forge prophet -y
pip install -r requirements.txt
```

---

## Alur kerja menjalankan proyek (ringkas)
1. Download data (contoh saham AAPL):
```powershell
python .\src\data\download_data.py --ticker AAPL --start 2015-01-01 --end 2024-01-01 --out data\raw
```

2. Preprocess:
```powershell
python .\src\preprocess.py --input data\raw\AAPL.csv --output data\processed\AAPL_parsed.csv
```

3. Feature engineering (lag & rolling):
- Gunakan `src/features/feature_engineering.py` di notebook atau script.

4. Backtesting & training (LightGBM baseline, rolling-origin):
```powershell
python .\src\evaluation\backtest.py --input data\processed\AAPL_parsed.csv
```
Hasil: `backtest_results.csv`, model di `models/`, dan metrics per window.

5. Buka JupyterLab untuk EDA dan modelling manual:
```powershell
jupyter lab
```
Buka `notebooks/01_EDA.ipynb` dan `notebooks/02_modeling.ipynb`.

6. Ekspor notebook menjadi HTML (untuk hasil mirip repo contoh):
```bash
jupyter nbconvert --to html notebooks/report.ipynb --output results/notebooks_html/report.html
```

---

## Artefak yang disarankan untuk hasil run
Simpan di folder `results/`:
- `<TICKER>_forecast.csv` (timestamp, actual, preds per model)
- `<TICKER>_metrics.csv` (model, window_end, MAE, RMSE, MAPE, sMAPE)
- `backtest_results.csv`
- `models/` (model files: .pkl, .h5)
- `diagnostics/` (residual plots, feature importance, SHAP)
- `notebooks_html/` (exported HTML reports)
- `run_info.json` (git commit, timestamp, env versions, args)

Checklist sebelum anggap "hasil akurat":
- Baseline naive dibandingkan
- Minimal 3 kelas model dibandingkan (statistik, ML, Prophet/NN)
- Rolling backtest dijalankan & metrik per window tersedia
- Interval prediksi & coverage dicatat
- Diagnostics & SHAP tersedia
- Notebook report di-export ke HTML

---

## Skrip PowerShell salin-tempel (Safe & One-Block)
Salin seluruh blok di bawah utuh dan paste ke PowerShell (sekaligus), lalu Enter. Jangan simpan sebagai .ps1 jika Anda ingin menghindari policy issues. Sesuaikan variabel di bagian atas bila perlu.

```powershell
# ----- START: Salin seluruh blok ini ke PowerShell lalu tekan Enter -----
# Konfigurasi (ubah sesuai kebutuhan)
$LocalPath  = "C:\Users\ASUS\Desktop\time-series-forecasting"
$RepoUrl    = "https://github.com/yirassssindaba-coder/Time-Series-Forecasting.git"
$UserName   = "Roberto Ocaviantyo Tahta Laksmana"
$UserEmail  = "yirassssindaba@gmail.com"
$CreateBasics = $true      # set $false jika tidak ingin membuat README/LICENSE/.gitignore
$ForcePush  = $false       # set $true hanya jika Anda paham konsekuensinya

function Write-Log { param($m,$level="INFO") $t=(Get-Date).ToString("yyyy-MM-dd HH:mm:ss"); Write-Host "[$t] [$level] $m" }

# Pastikan path lokal ada
if (-not (Test-Path -LiteralPath $LocalPath)) {
  Write-Log "Folder tidak ada, membuat: $LocalPath" "WARN"
  New-Item -ItemType Directory -Path $LocalPath -Force | Out-Null
}
Set-Location -LiteralPath $LocalPath

# Buat file dasar bila diminta
if ($CreateBasics) {
  if (-not (Test-Path README.md)) {
    @"
# Time-Series-Forecasting

Project: pipeline end-to-end untuk meramal deret waktu.
Author: $UserName
Contact: $UserEmail
"@ | Out-File -FilePath README.md -Encoding UTF8
    Write-Log "README.md dibuat."
  } else { Write-Log "README.md sudah ada." }

  if (-not (Test-Path LICENSE)) {
    @"
MIT License

Copyright (c) $(Get-Date -Format yyyy) $UserName

Permission is hereby granted, free of charge, to any person obtaining a copy...
"@ | Out-File -FilePath LICENSE -Encoding UTF8
    Write-Log "LICENSE dibuat."
  } else { Write-Log "LICENSE sudah ada." }

  if (-not (Test-Path .gitignore)) {
    @"
# Data and models
data/
models/
.env

# Python artifacts
__pycache__/
*.pyc
.venv/

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db
"@ | Out-File -FilePath .gitignore -Encoding UTF8
    Write-Log ".gitignore dibuat."
  } else { Write-Log ".gitignore sudah ada." }
}

# Helper untuk menjalankan git via cmd.exe /c
function Run-GitCmd {
  param([string] $cmdStr)
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    return @{ ExitCode = 1; Output = "git not found in PATH" }
  }
  $full = "git " + $cmdStr
  Write-Log "EXEC: $full" "DEBUG"
  $raw = & cmd.exe /c $full 2>&1
  $exit = $LASTEXITCODE
  if ($null -eq $raw) { $outStr = "" } elseif ($raw -is [System.Array]) { $outStr = ($raw -join "`n") } else { $outStr = [string]$raw }
  $outStr = ($outStr -replace "`r","").Trim()
  return @{ ExitCode = $exit; Output = $outStr }
}

# Pastikan git ada
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Write-Log "git tidak ditemukan di PATH. Install git lalu ulangi." "ERROR"
  return
}

# Inisialisasi git jika perlu
if (-not (Test-Path .git)) {
  $r = Run-GitCmd "init"
  if ($r.ExitCode -ne 0) { Write-Log "git init gagal:`n$r.Output" "ERROR"; return } else { Write-Log "git init berhasil." }
} else { Write-Log "Repository Git sudah ada." }

# Set git user lokal untuk repo ini
if ($UserName -and $UserEmail) {
  Run-GitCmd "config user.name `"$UserName`"" | Out-Null
  Run-GitCmd "config user.email `"$UserEmail`"" | Out-Null
  Write-Log "git user.name & user.email diset lokal." "INFO"
}

# Add all
$rAdd = Run-GitCmd "add ."
if ($rAdd.ExitCode -ne 0) { Write-Log "git add gagal:`n$rAdd.Output" "ERROR"; return } else { Write-Log "git add selesai." }

# Commit jika ada perubahan
$status = Run-GitCmd "status --porcelain"
if ([string]::IsNullOrWhiteSpace($status.Output)) {
  Write-Log "Tidak ada perubahan untuk di-commit." "INFO"
} else {
  $hasHead = (Run-GitCmd "rev-parse --verify HEAD").ExitCode -eq 0
  $msg = $hasHead ? "Update: commit full project" : "Initial commit: add full project"
  $rCommit = Run-GitCmd "commit -m `"$msg`""
  if ($rCommit.ExitCode -ne 0) { Write-Log "git commit bermasalah:`n$rCommit.Output" "WARN" } else { Write-Log "git commit berhasil." "INFO" }
}

# Pastikan branch main
Run-GitCmd "branch -M main" | Out-Null

# Atur remote origin
$remote = Run-GitCmd "remote get-url origin"
if ($remote.ExitCode -eq 0) {
  if ($remote.Output -ne $RepoUrl) {
    Run-GitCmd "remote set-url origin `"$RepoUrl`"" | Out-Null
    Write-Log "Remote origin diubah ke $RepoUrl" "INFO"
  } else { Write-Log "Remote origin sudah sesuai." "INFO" }
} else {
  $rRem = Run-GitCmd "remote add origin `"$RepoUrl`""
  if ($rRem.ExitCode -ne 0) { Write-Log "git remote add gagal:`n$rRem.Output" "ERROR"; return } else { Write-Log "Remote origin ditambahkan." }
}

# Push ke origin main
Write-Log "Melakukan push ke origin main..." "INFO"
$push = Run-GitCmd "push -u origin main"
if ($push.ExitCode -eq 0) { Write-Log "Push berhasil." "INFO"; if ($push.Output) { Write-Host $push.Output } }
else {
  Write-Log "Push gagal:`n$($push.Output)" "WARN"
  if ($push.Output -match "non-fast-forward" -or $push.Output -match "rejected") {
    Write-Log "Remote memiliki riwayat berbeda." "WARN"
    if ($ForcePush) {
      Write-Log "Melakukan push --force (ForcePush true)." "WARN"
      $f = Run-GitCmd "push -u origin main --force"
      if ($f.ExitCode -eq 0) { Write-Log "Force push berhasil." "INFO" } else { Write-Log "Force push gagal:`n$f.Output" "ERROR" }
    } else {
      Write-Host ""
      Write-Host "Untuk mengatasi: Ketik P untuk pull+merge, F untuk force push, atau Enter untuk batal."
      $choice = Read-Host "Pilihan (P/F/Enter)"
      if ($choice -match '^[Pp]') {
        $pull = Run-GitCmd "pull origin main --allow-unrelated-histories"
        if ($pull.ExitCode -ne 0) { Write-Log "Pull gagal:`n$($pull.Output)" "ERROR" } else {
          Write-Log "Pull berhasil, mencoba push lagi..." "INFO"
          $push2 = Run-GitCmd "push -u origin main"
          if ($push2.ExitCode -eq 0) { Write-Log "Push setelah pull berhasil." "INFO" } else { Write-Log "Push setelah pull gagal:`n$($push2.Output)" "ERROR" }
        }
      } elseif ($choice -match '^[Ff]') {
        $f = Run-GitCmd "push -u origin main --force"
        if ($f.ExitCode -eq 0) { Write-Log "Force push berhasil." "INFO" } else { Write-Log "Force push gagal:`n$f.Output" "ERROR" }
      } else { Write-Log "Operasi dibatalkan." "INFO" }
    }
  } else { Write-Log "Push gagal karena alasan lain. Periksa autentikasi/koneksi." "ERROR"; Write-Host $push.Output }
}

# Simpan ringkasan log
$log = @"
Timestamp: $(Get-Date -Format o)
LocalPath: $LocalPath
RepoUrl: $RepoUrl
GitStatus:
$(Run-GitCmd "status -s").Output

Remote:
$(Run-GitCmd "remote -v").Output
"@
$logPath = Join-Path $LocalPath "auto_push_summary.txt"
$log | Out-File -FilePath $logPath -Encoding UTF8
Write-Log "Ringkasan log disimpan: $logPath" "INFO"
Write-Log "Selesai." "INFO"
# ----- END blok -----
```

---

## Troubleshooting umum
- Error "git not found": install Git dan restart PowerShell.
- Error install Prophet di Windows: gunakan conda-forge (`conda install -c conda-forge prophet`).
- Error TensorFlow + CUDA: gunakan versi TensorFlow CPU atau pasang CUDA/cuDNN yang cocok.
- Jika `git push` ditolak karena non-fast-forward: gunakan opsi pull+merge dulu atau `--force` (perhatian: overwrite remote).

---

## Tips reproduksibilitas & kualitas
- Simpan `requirements.txt` atau `environment.yml`.
- Set random seed di semua library (numpy, random, tensorflow).
- Simpan `run_info.json` (commit SHA, paket versi, command args).
- Ekspor notebook final ke HTML dan simpan di `results/notebooks_html/`.

---

## Kontak
Roberto Ocaviantyo Tahta Laksmana  
yirassssindaba@gmail.com  
GitHub: https://github.com/yirassssindaba-coder

---

Terima kasih — README ini menggabungkan instruksi setup, alur kerja pipeline, artefak hasil, tips reproducibility, dan skrip PowerShell one-block yang aman untuk disalin langsung ke PowerShell. Jika Anda ingin, saya bisa:
- Sederhanakan skrip untuk hanya commit & push tanpa membuat file;
- Tambahkan fitur otomatis untuk mengecualikan file besar sebelum `git add`;
- Buat skrip terpisah untuk generate `results/` (backtest & export HTML).