@echo off
chcp 65001 >nul
cd /d "%~dp0"
where pwsh.exe >nul 2>nul
if errorlevel 1 (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -OpenBrowser
) else (
    pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -OpenBrowser
)
if errorlevel 1 (
    echo.
    echo TraceRAG 未能启动，请保留上方错误信息并查看 docs\daily-use.md。
    pause
)
