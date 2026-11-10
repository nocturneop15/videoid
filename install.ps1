# VideoID one-line installer:
#   irm https://raw.githubusercontent.com/nocturneop15/videoid/main/install.ps1 | iex
# Optional: $env:VIDEOID_DIR = "D:\Tools\VideoID" before running (default: %LOCALAPPDATA%\VideoID).
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Dir = if ($env:VIDEOID_DIR) { $env:VIDEOID_DIR } else { Join-Path $env:LOCALAPPDATA "VideoID" }
$Zip = Join-Path $env:TEMP "videoid-main.zip"
$Tmp = Join-Path $env:TEMP "videoid-main-extract"

Write-Host "Downloading VideoID into $Dir ..."
Invoke-WebRequest "https://github.com/nocturneop15/videoid/archive/refs/heads/main.zip" -OutFile $Zip
if (Test-Path $Tmp) { Remove-Item -Recurse -Force $Tmp }
Expand-Archive $Zip $Tmp -Force
New-Item -ItemType Directory -Force $Dir | Out-Null
# update the code, keep runtime\, videoid.db and config.json of an existing install
Copy-Item -Recurse -Force (Join-Path $Tmp "videoid-main\*") $Dir
Remove-Item -Recurse -Force $Tmp, $Zip

& (Join-Path $Dir "setup.ps1")

# put videoid.cmd on the user PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $Dir) {
    [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $Dir).TrimStart(";"), "User")
    $env:Path += ";$Dir"
    Write-Host "Added $Dir to your PATH (new terminals)."
}
Write-Host "Installed. Try:  videoid doctor   /   videoid scan D:\Movies"
