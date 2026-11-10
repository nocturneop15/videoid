@echo off
setlocal
set "PY=%~dp0runtime\python\python.exe"
if not exist "%~dp0runtime\.ready" (
    echo First run: downloading portable Python, Node.js, OCR engine into "%~dp0runtime" ...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" || exit /b 1
)
set PYTHONUTF8=1
"%PY%" -m videoid %*
