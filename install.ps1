# VideoID one-line installer:
#   irm https://raw.githubusercontent.com/nocturneop15/videoid/main/install.ps1 | iex
# Asks for the installation folder (Enter = %LOCALAPPDATA%\VideoID).
# Unattended: set $env:VIDEOID_DIR = "D:\Tools\VideoID" before running and no question is asked.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Default = Join-Path $env:LOCALAPPDATA "VideoID"
if ($env:VIDEOID_DIR) {
    $Dir = $env:VIDEOID_DIR
} else {
    while ($true) {
        $answer = (Read-Host "Install VideoID to [$Default]").Trim().Trim('"')
        $Dir = if ($answer) { [Environment]::ExpandEnvironmentVariables($answer) } else { $Default }
        $Dir = [IO.Path]::GetFullPath($Dir)
        $isInstall = Test-Path (Join-Path $Dir "videoid.cmd")
        $hasFiles = (Test-Path $Dir) -and (Get-ChildItem -Force $Dir | Select-Object -First 1)
        if ($isInstall) { Write-Host "Existing installation found - updating it (database, config and runtime are kept)."; break }
        if (-not $hasFiles) { break }
        $ok = Read-Host "$Dir is not empty. Install into it anyway? [y/N]"
        if ($ok -match '^(y|yes|a|ano)$') { break }
    }
}
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
