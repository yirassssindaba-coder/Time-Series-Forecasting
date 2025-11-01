# deploy-clean-fixed

Deskripsi singkat  
Skrip PowerShell ini menyinkronkan direktori proyek ke remote origin dengan alur aman add → commit → pull --rebase → push. Perbaikan utama pada versi ini adalah sanitasi daftar argumen untuk pemanggilan Start-Process dan penyediaan WorkingDirectory saat menjalankan git. Perbaikan tersebut menghindari kesalahan Start-Process terkait ArgumentList yang kosong.

Kebutuhan sistem  
1. PowerShell Core atau Windows PowerShell yang mendukung Start-Process  
2. Git tersedia di PATH sistem

Cara penggunaan  
1. Buka terminal PowerShell.  
2. Contoh menjalankan skrip:
   pwsh -ExecutionPolicy Bypass -File .\deploy-clean-fixed.ps1 -RepoDir "C:\Path\To\Project" -RemoteUrl "https://github.com/username/repo.git" -AutoSetRemote

Parameter  
1. RepoDir  
   Jenis: string  
   Nilai default: direktori kerja saat ini  
   Kegunaan: lokasi proyek yang akan disinkronkan

2. RemoteUrl  
   Jenis: string  
   Nilai default: kosong  
   Kegunaan: alamat remote origin yang akan ditambahkan atau diupdate jika diperlukan

3. Branch  
   Jenis: string  
   Nilai default: main  
   Kegunaan: nama cabang target push dan pull

4. AutoSetRemote  
   Jenis: switch  
   Kegunaan: jika diaktifkan dan RemoteUrl diberikan, origin akan diupdate otomatis ketika berbeda

5. ForcePush  
   Jenis: switch  
   Kegunaan: jika diaktifkan, push menggunakan force-with-lease untuk menimpa remote jika diperlukan

6. CommitMessage  
   Jenis: string  
   Nilai default: Add project files  
   Kegunaan: pesan commit yang digunakan saat membuat commit jika ada perubahan

7. VerboseLogs  
   Jenis: switch  
   Kegunaan: tampilkan log lebih detail

Ringkasan alur kerja skrip  
1. Memeriksa ketersediaan git  
2. Meresolve dan berpindah ke direktori proyek  
3. Membuat file dasar jika belum ada: README.md, .gitignore, LICENSE  
4. Menginisialisasi repositori git jika belum ada  
5. Menambahkan semua perubahan dan membuat commit bila diperlukan  
6. Memastikan nama cabang sesuai parameter Branch  
7. Memeriksa dan menambahkan atau memperbarui origin jika RemoteUrl diberikan dan AutoSetRemote diizinkan  
8. Melakukan fetch lalu pull dengan metode rebase  
9. Melakukan push ke origin pada cabang yang ditentukan  
10. Memberikan petunjuk aman apabila terjadi konflik atau penolakan push

Blok skrip lengkap  
Skrip berikut dimasukkan persis seperti file deploy-clean-fixed.ps1. Blok skrip diawali dan diakhiri dengan lima tanda kutip tunggal sesuai permintaan:

'''''
param(
  [string] $RepoDir = (Get-Location).ProviderPath,
  [string] $RemoteUrl = "",
  [string] $Branch = "main",
  [switch] $AutoSetRemote,
  [switch] $ForcePush,
  [string] $CommitMessage = "Add project files",
  [switch] $VerboseLogs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log {
  param([string]$Msg, [string]$Level="INFO")
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  switch ($Level) {
    "ERROR" { Write-Host "[$ts] [$Level] $Msg" -ForegroundColor Red; break }
    "WARN"  { Write-Host "[$ts] [$Level] $Msg" -ForegroundColor Yellow; break }
    default {
      if ($VerboseLogs) { Write-Host "[$ts] [$Level] $Msg" -ForegroundColor Cyan }
      else { Write-Host "[$ts] [$Level] $Msg" }
    }
  }
}

function Run-GitProcess {
  param(
    [Parameter(Mandatory=$true)][string[]] $Args,
    [string] $WorkingDirectory = (Get-Location).Path
  )

  # Sanitize args: remove $null and empty-string elements, convert to strings
  $safeArgs = @()
  if ($null -ne $Args) {
    foreach ($a in $Args) {
      if ($null -ne $a) {
        $s = [string]$a
        if (-not [string]::IsNullOrEmpty($s)) { $safeArgs += $s }
      }
    }
  }

  # If caller passed empty array after sanitization, supply a benign default arg to avoid Start-Process error.
  # Use "--version" only when nothing meaningful provided (should not normally happen).
  if ($safeArgs.Count -eq 0) { $safeArgs = @("--version") }

  # Prepare temporary files for stdout/stderr
  $outFile = [System.IO.Path]::GetTempFileName()
  $errFile = [System.IO.Path]::GetTempFileName()

  try {
    # Use Start-Process with explicit WorkingDirectory to ensure git runs in correct repo context
    $psi = Start-Process -FilePath "git" -ArgumentList $safeArgs -NoNewWindow -Wait -PassThru `
           -RedirectStandardOutput $outFile -RedirectStandardError $errFile -WorkingDirectory $WorkingDirectory

    $exit = $psi.ExitCode
    $out = ""
    $err = ""
    try { $out = Get-Content -Raw -LiteralPath $outFile -ErrorAction SilentlyContinue } catch {}
    try { $err = Get-Content -Raw -LiteralPath $errFile -ErrorAction SilentlyContinue } catch {}
    $combined = ($out + "`n" + $err).Trim()
    return @{ ExitCode = $exit; Output = $combined }
  } catch {
    # Return error-like structure rather than throwing to the caller
    $msg = $_.Exception.Message
    return @{ ExitCode = 1; Output = "Start-Process/git failed: $msg" }
  } finally {
    Remove-Item -LiteralPath $outFile -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $errFile -ErrorAction SilentlyContinue
  }
}

function Get-ExistingRemoteFromRemoteV {
  $r = Run-GitProcess @("remote","-v") (Get-Location).Path
  if ($r.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($r.Output)) { return $null }
  foreach ($line in $r.Output -split "`n") {
    $parts = $line -split "\s+"
    if ($parts.Count -ge 2 -and $parts[0] -eq "origin" -and $line -match "\((fetch|push)\)$") { return $parts[1].Trim() }
  }
  return $null
}

function Ensure-FileIfMissing {
  param([string]$Path, [string]$Contents)
  if (-not (Test-Path -LiteralPath $Path)) {
    $dir = Split-Path -Path $Path -Parent
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Set-Content -Path $Path -Value $Contents -Encoding UTF8
    Write-Log "Created $Path" "INFO"
  } else { Write-Log "$Path already exists" "INFO" }
}

# Begin
Write-Log "Starting deploy-clean-fixed synchronization..." "INFO"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Write-Log "git not found in PATH. Install Git first." "ERROR"; exit 1
}

try { $RepoDir = (Resolve-Path -LiteralPath $RepoDir -ErrorAction Stop).ProviderPath } catch { Write-Log "RepoDir not found: $RepoDir" "ERROR"; exit 1 }
Set-Location -LiteralPath $RepoDir
Write-Log "Operating in: $RepoDir" "INFO"

# Create basic files if missing
Ensure-FileIfMissing -Path (Join-Path $RepoDir "README.md") -Contents "# Project"
Ensure-FileIfMissing -Path (Join-Path $RepoDir ".gitignore") -Contents @'
__pycache__/
.venv/
venv/
*.py[cod]
.ipynb_checkpoints/
.env
'@
Ensure-FileIfMissing -Path (Join-Path $RepoDir "LICENSE") -Contents "MIT License - add your license text"

# git init if needed
if (-not (Test-Path -LiteralPath (Join-Path $RepoDir ".git"))) {
  Write-Log "No .git found. Initializing repository..." "INFO"
  $init = Run-GitProcess @("init") $RepoDir
  if ($init.ExitCode -ne 0) { Write-Log "git init failed: $($init.Output)" "ERROR"; exit 1 }
  Write-Log "Repository initialized." "INFO"
} else { Write-Log ".git found - repository already initialized." "INFO" }

# git add --all
Write-Log "Running git add --all" "INFO"
$add = Run-GitProcess @("add","--all") $RepoDir
if ($add.ExitCode -ne 0) { Write-Log "git add returned warnings/errors: $($add.Output)" "WARN" } else { Write-Log "git add OK" "INFO" }

# commit if changes
$status = Run-GitProcess @("status","--porcelain") $RepoDir
if (-not [string]::IsNullOrWhiteSpace($status.Output)) {
  Write-Log "Changes detected - creating commit" "INFO"
  $commit = Run-GitProcess @("commit","-m",$CommitMessage) $RepoDir
  if ($commit.ExitCode -ne 0) {
    Write-Log "Commit failed: $($commit.Output)" "WARN"
    # try set local user config fallback
    $cfgName = Run-GitProcess @("config","--get","user.name") $RepoDir
    $cfgEmail = Run-GitProcess @("config","--get","user.email") $RepoDir
    if ([string]::IsNullOrWhiteSpace($cfgName.Output) -or [string]::IsNullOrWhiteSpace($cfgEmail.Output)) {
      Write-Log "Setting temporary local git user.name/email" "INFO"
      Run-GitProcess @("config","user.name","auto-committer") $RepoDir | Out-Null
      Run-GitProcess @("config","user.email","auto@local") $RepoDir | Out-Null
      $commit2 = Run-GitProcess @("commit","-m",$CommitMessage) $RepoDir
      if ($commit2.ExitCode -ne 0) { Write-Log "Commit retry failed: $($commit2.Output)" "ERROR"; exit 1 } else { Write-Log "Commit created after fallback." "INFO" }
    } else {
      Write-Log "Commit failed but user config present. Inspect above output." "ERROR"; exit 1
    }
  } else { Write-Log "Commit created." "INFO" }
} else { Write-Log "No changes to commit." "INFO" }

# ensure branch main
$curBranchRes = Run-GitProcess @("rev-parse","--abbrev-ref","HEAD") $RepoDir
if ($curBranchRes.ExitCode -ne 0) { Write-Log "Could not determine current branch: $($curBranchRes.Output)" "WARN" }
else {
  $currentBranch = $curBranchRes.Output.Trim()
  if ($currentBranch -ne $Branch) {
    Write-Log "Renaming current branch ($currentBranch) to $Branch" "INFO"
    $br = Run-GitProcess @("branch","-M",$Branch) $RepoDir
    if ($br.ExitCode -ne 0) { Write-Log "Failed to rename branch: $($br.Output)" "ERROR"; exit 1 }
  } else { Write-Log "Already on branch $Branch" "INFO" }
}

# parse existing remote from `git remote -v`
$existingRemote = $null
$remoteV = Run-GitProcess @("remote","-v") $RepoDir
if ($remoteV.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($remoteV.Output)) {
  foreach ($l in $remoteV.Output -split "`n") {
    $parts = $l -split "\s+"
    if ($parts.Count -ge 2 -and $parts[0] -eq "origin" -and $l -match "\((fetch|push)\)$") { $existingRemote = $parts[1].Trim(); break }
  }
}

if (-not $existingRemote) {
  if ($RemoteUrl) {
    Write-Log "No origin configured. Adding origin -> $RemoteUrl" "INFO"
    $r = Run-GitProcess @("remote","add","origin",$RemoteUrl) $RepoDir
    if ($r.ExitCode -ne 0) { Write-Log "git remote add failed: $($r.Output)" "ERROR"; exit 1 }
    Write-Log "Origin set." "INFO"
  } else {
    Write-Log "No origin remote configured and no RemoteUrl provided. Skipping push steps." "WARN"
    Write-Log "Local changes committed." "INFO"
    exit 0
  }
} else {
  Write-Log "Existing origin: $existingRemote" "INFO"
  if ($RemoteUrl -and ($existingRemote -ne $RemoteUrl)) {
    if ($AutoSetRemote) {
      Write-Log "Updating origin URL to $RemoteUrl (AutoSetRemote enabled)" "INFO"
      $r = Run-GitProcess @("remote","set-url","origin",$RemoteUrl) $RepoDir
      if ($r.ExitCode -ne 0) { Write-Log "Failed to set remote URL: $($r.Output)" "ERROR"; exit 1 }
      Write-Log "Origin URL updated." "INFO"
    } else {
      Write-Log "Remote URL differs from provided RemoteUrl. Use -AutoSetRemote to update it automatically." "WARN"
    }
  }
}

# fetch & pull --rebase
Write-Log "Fetching origin..." "INFO"
$fetch = Run-GitProcess @("fetch","origin") $RepoDir
if ($fetch.ExitCode -ne 0) { Write-Log "git fetch returned: $($fetch.Output)" "WARN" }

Write-Log "Attempting git pull --rebase origin $Branch" "INFO"
$pull = Run-GitProcess @("pull","--rebase","origin",$Branch) $RepoDir
if ($pull.ExitCode -ne 0) {
  Write-Log "Pull (rebase) returned non-zero: $($pull.Output)" "ERROR"
  Write-Log "If conflicts occurred: edit files, then run:" "INFO"
  Write-Host '  git add "<resolved-file>"'
  Write-Host '  git rebase --continue'
  Write-Host 'Or abort with: git rebase --abort'
  Write-Log "After resolving rebase, run: git push origin $Branch" "INFO"
  exit 1
} else {
  Write-Log "Pull --rebase succeeded." "INFO"
}

# push
Write-Log "Pushing to origin/$Branch..." "INFO"
if ($ForcePush) {
  $push = Run-GitProcess @("push","--force-with-lease","-u","origin",$Branch) $RepoDir
} else {
  $push = Run-GitProcess @("push","-u","origin",$Branch) $RepoDir
}
if ($push.ExitCode -ne 0) {
  Write-Log "git push failed: $($push.Output)" "ERROR"
  if ($push.Output -match "non-fast-forward" -or $push.Output -match "rejected" -or $push.Output -match "fetch first") {
    Write-Log "Push rejected (non-fast-forward). Recommended safe steps:" "WARN"
    Write-Host '  git fetch origin'
    Write-Host "  git pull --rebase origin $Branch"
    Write-Host '  resolve conflicts -> git add "<resolved-file>" -> git rebase --continue'
    Write-Host "  git push origin $Branch"
    Write-Host ''
    Write-Host "If you must overwrite remote history, re-run script with -ForcePush (dangerous)."
  }
  exit 1
} else {
  Write-Log "Push succeeded." "INFO"
}

Write-Log "All done — repository synchronized." "INFO"
'''''

Catatan akhir  
Skrip dimasukkan persis seperti file aslinya. Jika Anda ingin format blok pembuka/penutup berbeda atau menambahkan penjelasan lebih ringkas di bagian atas readme, beri tahu saya dan saya akan perbarui.
