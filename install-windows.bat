@echo off
rem ============================================================================
rem  OllaVoice - one-click Windows installer
rem
rem  Double-click this file, or run it from a terminal:
rem      install-windows.bat
rem      install-windows.bat -Tier high -Model llama3.1:8b
rem
rem  All arguments are passed straight through to scripts\setup.ps1
rem  (run "install-windows.bat -help" for the full list).
rem ============================================================================
setlocal
cd /d "%~dp0"

where powershell >nul 2>nul
if errorlevel 1 (
    echo ERROR: PowerShell was not found. This installer needs Windows PowerShell 5.1+
    echo        ^(included with Windows 10 and 11^).
    pause
    exit /b 1
)

echo.
echo  OllaVoice installer - this takes a few minutes the first time.
echo  It creates .venv, installs the local speech models, and adds Start Menu shortcuts.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
    echo  Finished. Start it from the Start Menu, or run:
    echo      .venv\Scripts\ollavoice.exe gui
) else (
    echo  The installer reported warnings ^(exit code %RC%^). Scroll up for details.
    echo  You can re-run it at any time - it skips work that is already done.
)
echo.
pause
exit /b %RC%
