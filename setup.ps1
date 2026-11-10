# VideoID portable runtime installer (run automatically by videoid.cmd on first start).
# Creates .\runtime with embedded CPython, portable Node.js and all dependencies; nothing is installed system-wide.
# Re-run with -Force to rebuild the runtime from scratch.
param(
    [string]$PythonVersion = "3.13.9",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root    = $PSScriptRoot
$Runtime = Join-Path $Root "runtime"
$Py      = Join-Path $Runtime "python"
$Exe     = Join-Path $Py "python.exe"
$Ready   = Join-Path $Runtime ".ready"

if ($Force -and (Test-Path $Runtime)) { Remove-Item -Recurse -Force $Runtime }
if (Test-Path $Ready) { Remove-Item $Ready }

# ---- embedded Python ----
if (-not (Test-Path $Exe)) {
    New-Item -ItemType Directory -Force $Py | Out-Null
    $zip = Join-Path $env:TEMP "python-$PythonVersion-embed-amd64.zip"
    Write-Host "Downloading embedded Python $PythonVersion ..."
    Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $Py -Force
    Remove-Item $zip

    # Enable site-packages and put the tool root (..\..) on sys.path - relative, so the folder stays portable.
    $pth = Get-ChildItem $Py -Filter "python*._pth" | Select-Object -First 1
    $stdzip = (Get-ChildItem $Py -Filter "python*.zip" | Select-Object -First 1).Name
    Set-Content -Encoding ascii -Path $pth.FullName -Value @($stdzip, ".", "Lib\site-packages", "..\..", "import site")

    Write-Host "Installing pip ..."
    $gp = Join-Path $env:TEMP "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $gp
    & $Exe $gp --no-warn-script-location --no-cache-dir -q
    Remove-Item $gp
}

Write-Host "Installing Python packages ..."
& $Exe -m pip install -q --no-cache-dir --no-warn-script-location --upgrade -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE) { throw "pip install failed" }
# sympy/mpmath (~70 MB) are only used by onnxruntime's developer tools, not for inference
& $Exe -m pip uninstall -q -y sympy mpmath 2>$null

# ---- portable Node.js (LTS) + node-csfd-api for CSFD access ----
$NodeDir = Join-Path $Runtime "node"
if (-not (Test-Path (Join-Path $NodeDir "node.exe"))) {
    $idx = Invoke-RestMethod "https://nodejs.org/dist/index.json"
    $lts = ($idx | Where-Object { $_.lts -and $_.files -contains "win-x64-zip" } | Select-Object -First 1).version
    Write-Host "Downloading portable Node.js $lts ..."
    $nz = Join-Path $env:TEMP "node-$lts-win-x64.zip"
    Invoke-WebRequest -Uri "https://nodejs.org/dist/$lts/node-$lts-win-x64.zip" -OutFile $nz
    $tmp = Join-Path $env:TEMP "node-extract-$lts"
    Expand-Archive -Path $nz -DestinationPath $tmp -Force
    Move-Item (Join-Path $tmp "node-$lts-win-x64") $NodeDir
    Remove-Item $nz; Remove-Item -Recurse $tmp
    # not needed at runtime
    foreach ($d in "node_modules\corepack", "node_modules\npm\docs", "node_modules\npm\man") {
        Remove-Item -Recurse -Force (Join-Path $NodeDir $d) -ErrorAction SilentlyContinue
    }
}
$CsfdDir = Join-Path $Runtime "csfd"
New-Item -ItemType Directory -Force $CsfdDir | Out-Null
if (-not (Test-Path (Join-Path $CsfdDir "package.json"))) {
    Set-Content -Encoding ascii (Join-Path $CsfdDir "package.json") '{ "name": "videoid-csfd", "private": true, "type": "module" }'
}
Write-Host "Installing node-csfd-api ..."
$env:PATH = "$NodeDir;$env:PATH"
$env:npm_config_cache = Join-Path $env:TEMP "videoid-npm-cache"
Push-Location $CsfdDir
& (Join-Path $NodeDir "npm.cmd") install --no-audit --no-fund --omit=dev --loglevel=error node-csfd-api@latest
$rc = $LASTEXITCODE
Pop-Location
Remove-Item -Recurse -Force $env:npm_config_cache -ErrorAction SilentlyContinue
if ($rc) { throw "npm install node-csfd-api failed" }

# ---- OCR models ----
# Only the default Latin models now; other languages (--lang hi,ja,...) download on first use (~8 MB each).
Write-Host "Fetching OCR models ..."
Push-Location $Root
& $Exe -c "from videoid.ocr import OCR; import numpy as np; OCR({'languages': ['latin']}).run(np.zeros((64,256,3),np.uint8))"
$rc = $LASTEXITCODE
Pop-Location
if ($rc) { throw "OCR model download failed" }

Set-Content -Encoding ascii $Ready (Get-Date -Format s)
Write-Host "Runtime ready."
