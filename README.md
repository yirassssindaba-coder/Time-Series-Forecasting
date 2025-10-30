# Panduan Push Proyek ke GitHub (PowerShell)

Panduan PowerShell langkah demi langkah untuk meng‑commit seluruh folder lokal
C:\Users\ASUS\Desktop\time-series-forecasting ke repository:
https://github.com/yirassssindaba-coder/Time-Series-Forecasting.git

Sebelum mulai: buka PowerShell (sebagai user biasa). Jika eksekusi skrip diblokir, jalankan:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
```

---

## A. Jika folder BELUM di‑inisialisasi sebagai Git repo (baru)

1. Pindah ke folder proyek:
```powershell
cd "C:\Users\ASUS\Desktop\time-series-forecasting"
```

2. (Opsional) Set nama/email untuk commit lokal:
```powershell
git config user.name "Nama Anda"
git config user.email "email@anda.com"
```

3. Inisialisasi repo, tambahkan semua file, dan commit:
```powershell
git init
git add .
git commit -m "Initial commit: add full project"
```

4. Set branch utama jadi `main`:
```powershell
git branch -M main
```

5. Tambahkan remote origin (HTTPS) dan push:
```powershell
git remote add origin https://github.com/yirassssindaba-coder/Time-Series-Forecasting.git
git push -u origin main
```

- Jika ingin pakai SSH (pastikan SSH key sudah terdaftar di GitHub):
```powershell
git remote add origin git@github.com:yirassssindaba-coder/Time-Series-Forecasting.git
git push -u origin main
```

---

## B. Jika folder SUDAH berisi repo Git lokal (ada folder .git)

1. Masuk folder:
```powershell
cd "C:\Users\ASUS\Desktop\time-series-forecasting"
```

2. Cek status & remote:
```powershell
git status
git remote -v
```

3a. Jika *remote origin* belum diset atau ingin mengganti:
```powershell
git remote remove origin   # hanya jika ingin ganti remote
git remote add origin https://github.com/yirassssindaba-coder/Time-Series-Forecasting.git
```

3b. Jika remote ada tetapi ingin ubah URL:
```powershell
git remote set-url origin https://github.com/yirassssindaba-coder/Time-Series-Forecasting.git
# atau untuk SSH:
git remote set-url origin git@github.com:yirassssindaba-coder/Time-Series-Forecasting.git
```

4. Tambah, commit, dan push:
```powershell
git add .
git commit -m "Update: commit full project"
git push -u origin main
```

---

## Jika push ditolak karena remote memiliki riwayat berbeda

- Opsi aman (pull & merge), lalu push:
```powershell
git pull origin main --allow-unrelated-histories
# Selesaikan merge conflicts jika muncul, lalu:
git add .
git commit -m "Resolve merge"
git push origin main
```

- Opsi memaksa (HATI‑HATI: menimpa remote):
```powershell
git push -u origin main --force
```

---

## Autentikasi: HTTPS vs SSH

- HTTPS: Anda mungkin diminta username/password — gunakan Personal Access Token (PAT) sebagai password. Disarankan pakai Git Credential Manager atau `gh`.
- SSH: Pastikan SSH key sudah dibuat dan ditambahkan ke akun GitHub. Tes koneksi:
```powershell
ssh -T git@github.com
```

---

## Tips penting sebelum push

- Pastikan `.gitignore` mengecualikan file besar (data/, models/, .env). Jika belum ada, buat:
```powershell
New-Item -Path .gitignore -ItemType File -Force
Add-Content .gitignore "data/"
Add-Content .gitignore "models/"
Add-Content .gitignore ".env"
Add-Content .gitignore "*.pyc"
```

- Jika sudah terlanjur menambahkan file besar, hapus dari index lalu commit:
```powershell
git rm --cached path\to\bigfile
git commit -m "Remove large file from history before push"
```

- Untuk membersihkan riwayat (history) lebih lanjut gunakan tool seperti BFG atau `git filter-repo`.

---

## Periksa hasil setelah push

```powershell
git remote -v
git log --oneline -n 5
git ls-remote origin
```

Setelah push sukses, buka repository di:
https://github.com/yirassssindaba-coder/Time-Series-Forecasting

---

## Saya bisa bantu lebih lanjut

Jika Anda mau, saya dapat:
- Menyediakan perintah PowerShell untuk membuat file README.md, LICENSE, dan .gitignore otomatis sebelum commit.
- Membuat skrip PowerShell `.ps1` lengkap yang menjalankan: inisialisasi, set remote, commit, dan push sehingga Anda tinggal menjalankannya sekali.

Jika ingin skrip otomatis, jalankan perintah berikut di PowerShell untuk membuat file skrip kosong lalu saya akan berikan isinya:
```powershell
New-Item -Path .\scripts -ItemType Directory -Force
New-Item -Path .\scripts\auto_push.ps1 -ItemType File -Force
notepad .\scripts\auto_push.ps1
```

---

## Catatan keamanan
- Hati‑hati saat menggunakan `--force`; ini dapat menimpa riwayat remote.
- Jangan commit kredensial atau file sensitif (.env, API keys).
