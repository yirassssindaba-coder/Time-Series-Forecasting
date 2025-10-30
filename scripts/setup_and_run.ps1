<#
.SYNOPSIS
  Setup environment (venv) dan install dependency dasar.

.USAGE
  PowerShell:
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
    .\scripts\setup_and_run.ps1

.NOTES
  Jika Anda memilih conda, aktifkan conda env secara manual lalu jalankan install requirements.
#>

param(
  [string] $Python = "python"
)

Write-Host "1) Membuat virtual environment .venv"
& $Python -m venv .venv

Write-Host "2) Mengaktifkan virtualenv"
# Aktifkan current session
. .\.venv\Scripts\Activate.ps1

Write-Host "3) Upgrade pip dan install dependency"
& $Python -m pip install --upgrade pip
# Install requirements (jika prophet bermasalah, gunakan conda)
if (Test-Path -Path "requirements.txt") {
  & $Python -m pip install -r requirements.txt
} else {
  Write-Warning "requirements.txt tidak ditemukan di working directory."
}

Write-Host "Selesai setup. Virtualenv aktif di sesi ini."
Write-Host "Untuk langkah demo: jalankan scripts\\run_demo.ps1 atau ikuti commands manual di README."