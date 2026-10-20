<#
Build the OpenOne Windows app: PyInstaller bundle, portable zip and Inno Setup installer.

  powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 [-Python py.exe] [-Iscc ISCC.exe] [-Offline]

  -Offline   also build "-offline" variants with the CPU inference runtime bundled (works without internet at
             onboarding; adds ~1 GB unpacked / ~300 MB compressed).
Outputs in release\: OpenOne-Setup-<ver>-win-x64.exe, OpenOne-<ver>-win-x64-portable.zip (+ -offline variants) and
SHA256SUMS.txt. Unsigned unless you sign the files afterwards (signtool) with your own certificate.
#>
param(
  [string]$Python = "python",
  [string]$Iscc = "",
  [switch]$Offline
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root
$Version = "0.1.0"
$Venv = Join-Path $Root ".build-venv"
$Release = Join-Path $Root "release"

if (-not (Test-Path "$Venv\Scripts\python.exe")) { & $Python -m venv $Venv }
& "$Venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r packaging\requirements-build.txt
if ($LASTEXITCODE) { throw "pip install failed" }

Remove-Item -Recurse -Force dist\OpenOne, build -ErrorAction SilentlyContinue
& "$Venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\openone.spec
if ($LASTEXITCODE) { throw "PyInstaller failed" }

# smoke test: the frozen app must start headless and answer /api/status
$tmp = Join-Path $env:TEMP ("openone-build-test-" + [guid]::NewGuid())
New-Item -ItemType Directory $tmp | Out-Null
$env:OPENONE_DATA_DIR = $tmp
$p = Start-Process -FilePath dist\OpenOne\OpenOne.exe -ArgumentList "--serve","0" -PassThru
try {
  for ($i = 0; $i -lt 60 -and -not (Test-Path "$tmp\serve-url.txt"); $i++) { Start-Sleep -Milliseconds 500 }
  $url = (Get-Content "$tmp\serve-url.txt" -Raw).Trim()
  $tok = ($url -split '\?t=')[1]
  $base = ($url -split '/\?')[0]
  $st = Invoke-RestMethod -Uri "$base/api/status" -Headers @{ "X-OpenOne-Token" = $tok }
  if ($st.app_version -ne $Version) { throw "smoke test: unexpected status" }
  Write-Host "smoke test ok: $base"
} finally {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  Remove-Item Env:\OPENONE_DATA_DIR
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

if (-not $Iscc) {
  foreach ($c in @("$Root\_tools\innosetup\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
    if (Test-Path $c) { $Iscc = $c; break }
  }
}
if (-not $Iscc) { throw "Inno Setup 6 (ISCC.exe) not found; pass -Iscc" }
New-Item -ItemType Directory -Force $Release | Out-Null

function Build-Variant([string]$suffix) {
  $zip = "$Release\OpenOne-$Version-win-x64$suffix-portable.zip"
  $stage = Join-Path $env:TEMP ("openone-portable-" + [guid]::NewGuid())
  New-Item -ItemType Directory "$stage\OpenOne" | Out-Null
  Copy-Item -Recurse dist\OpenOne\* "$stage\OpenOne"
  Set-Content -Path "$stage\OpenOne\portable.txt" -Value "Portable mode: models, runtimes, settings and logs are stored in the data folder next to OpenOne.exe. Delete this file to use %LOCALAPPDATA%\OpenOne instead." -Encoding utf8
  Remove-Item $zip -ErrorAction SilentlyContinue
  & "$Venv\Scripts\python.exe" -c "import shutil,sys; shutil.make_archive(sys.argv[1][:-4], 'zip', sys.argv[2], 'OpenOne')" $zip $stage
  Remove-Item -Recurse -Force $stage
  & $Iscc /Q "/DAppVersion=$Version" "/DDistDir=$Root\dist\OpenOne" "/DOutDir=$Release" "/DSuffix=$suffix" packaging\windows\openone.iss
  if ($LASTEXITCODE) { throw "ISCC failed" }
}

Build-Variant ""
if ($Offline) {
  & "$Venv\Scripts\python.exe" packaging\bundle_runtime.py cpu dist\OpenOne\runtime
  if ($LASTEXITCODE) { throw "runtime bundling failed" }
  Build-Variant "-offline"
  Remove-Item -Recurse -Force dist\OpenOne\runtime
}

Get-ChildItem $Release -File | Where-Object { $_.Name -ne "SHA256SUMS.txt" } | ForEach-Object {
  "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
} | Set-Content "$Release\SHA256SUMS.txt" -Encoding ascii
Get-ChildItem $Release | Format-Table Name, @{n="MB"; e={[math]::Round($_.Length / 1MB, 1)}}
