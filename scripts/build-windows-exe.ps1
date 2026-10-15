<#
.SYNOPSIS
    Build a standalone OllaVoice.exe for Windows with PyInstaller.

.DESCRIPTION
    Must be run ON Windows - PyInstaller cannot cross-compile. Output lands in
    dist\OllaVoice\ (a folder, not a single file: onedir starts in ~2 s whereas
    onefile has to unpack ~1.5 GB of native libraries on every launch).

    Speech model weights are NOT included. The built app reads them from the
    standard location (%LOCALAPPDATA%\OllaVoice\models), so distribute the exe
    folder together with either:
      * instructions to run `OllaVoice.exe` once on a machine with internet, or
      * an offline bundle made by `ollavoice bundle <folder>` (see
        docs\INSTALL-windows.md).

.PARAMETER OneFile
    Produce a single .exe instead of a folder. Slower to start, easier to copy.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build-windows-exe.ps1
#>
[CmdletBinding()]
param(
    [switch]$OneFile,
    [switch]$SkipTests,
    [string]$Venv = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ScriptDir

if ($env:OS -ne 'Windows_NT') {
    Write-Host 'This script must run on Windows: PyInstaller cannot cross-compile.' -ForegroundColor Red
    exit 1
}

if (-not $Venv) { $Venv = Join-Path $Root '.venv' }
$Py = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $Py)) {
    Write-Host "No virtual environment at $Venv." -ForegroundColor Yellow
    Write-Host 'Run scripts\setup.ps1 first, or pass -Venv "C:\path\to\venv".' -ForegroundColor Yellow
    exit 1
}

Push-Location $Root
try {
    Write-Host '==> Installing build dependencies' -ForegroundColor Cyan
    & $Py -m pip install --disable-pip-version-check --quiet pyinstaller
    if ($LASTEXITCODE -ne 0) { throw 'pyinstaller install failed' }

    if (-not $SkipTests) {
        Write-Host '==> Running the test suite' -ForegroundColor Cyan
        & $Py -m pip install --disable-pip-version-check --quiet pytest | Out-Null
        & $Py -m pytest tests -q -m 'not integration'
        if ($LASTEXITCODE -ne 0) { throw 'tests failed - not building' }
    }

    Write-Host '==> Cleaning previous builds' -ForegroundColor Cyan
    foreach ($Dir in @('build', 'dist')) {
        $Path = Join-Path $Root $Dir
        if (Test-Path $Path) { Remove-Item -Recurse -Force $Path }
    }

    Write-Host '==> Building with PyInstaller (this takes several minutes)' -ForegroundColor Cyan
    $Args = @('-m', 'PyInstaller', '--noconfirm', '--clean',
              'packaging\OllaVoice.spec')
    if ($OneFile) {
        # The spec builds onedir; for onefile we rebuild from the entry point.
        $Args = @('-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile',
                  '--windowed', '--name', 'OllaVoice',
                  '--collect-all', 'piper',
                  '--collect-all', 'espeakng_loader',
                  '--collect-all', 'ctranslate2',
                  '--collect-all', 'onnxruntime',
                  '--collect-all', 'faster_whisper',
                  '--collect-all', 'kokoro_onnx',
                  '--add-data', 'ollavoice\server\static;ollavoice\server\static',
                  '--exclude-module', 'tkinter',
                  '--exclude-module', 'matplotlib',
                  '--exclude-module', 'scipy',
                  '--exclude-module', 'pandas',
                  '--exclude-module', 'torch',
                  'run_gui.py')
    }
    & $Py @Args
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }

    $Out = Join-Path $Root 'dist\OllaVoice'
    Write-Host ''
    Write-Host '==> Smoke-testing the build' -ForegroundColor Cyan
    $Exe = Join-Path $Out 'OllaVoice.exe'
    if (-not (Test-Path $Exe)) { $Exe = Join-Path $Root 'dist\OllaVoice.exe' }
    if (Test-Path $Exe) {
        $Size = [math]::Round((Get-ChildItem $Exe).Length / 1MB, 1)
        Write-Host "    built $Exe ($Size MB)" -ForegroundColor Green
        if (Test-Path $Out) {
            $Total = [math]::Round(((Get-ChildItem $Out -Recurse | Measure-Object Length -Sum).Sum) / 1MB, 1)
            Write-Host "    dist\OllaVoice total: $Total MB" -ForegroundColor Green
        }
    } else {
        Write-Warning 'could not locate the built executable'
    }

    Write-Host ''
    Write-Host 'Next steps:' -ForegroundColor White
    Write-Host '  1. Copy dist\OllaVoice\ to the target PC.' -ForegroundColor DarkGray
    Write-Host '  2. Install Ollama there and pull a model:  ollama pull llama3.2:3b' -ForegroundColor DarkGray
    Write-Host '  3. Provide the speech models - either run OllaVoice once with internet,' -ForegroundColor DarkGray
    Write-Host '     or copy an offline bundle into %LOCALAPPDATA%\OllaVoice\models.' -ForegroundColor DarkGray
    Write-Host ''
}
finally {
    Pop-Location
}
