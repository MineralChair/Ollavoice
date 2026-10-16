<#
.SYNOPSIS
    One-shot Windows installer for OllaVoice (local speech input + output for Ollama).

.DESCRIPTION
    Creates a dedicated virtual environment next to this script, installs the Python
    packages, runs the setup wizard (hardware detection + speech model downloads),
    and adds Start Menu shortcuts.

    Everything it installs runs locally. The only network access happens here, during
    setup - after that you can unplug the machine and it keeps working.

.PARAMETER Tier
    low | balanced | high. Omitted = auto-detect from your hardware.

.PARAMETER NoGui
    Terminal-only install (skips PySide6, ~600 MB smaller).

.PARAMETER Offline
    Install from .\vendor\wheels and .\vendor\models without touching the network.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Tier high -Model llama3.1:8b
#>
[CmdletBinding()]
param(
    [ValidateSet('low', 'balanced', 'high', 'custom')]
    [string]$Tier = '',
    [string]$Model = '',
    [string]$Language = '',
    [string]$TtsEngine = '',
    [string]$TtsVoice = '',
    [string]$SttModel = '',
    [string]$PythonExe = '',
    [switch]$NoGui,
    [switch]$NoTts,
    [switch]$NoStt,
    [switch]$NoPull,
    [switch]$Pull,
    [switch]$Offline,
    [switch]$NoShortcuts,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ScriptDir
$Venv      = Join-Path $Root '.venv'
$Py        = Join-Path $Venv 'Scripts\python.exe'
$Vendor    = Join-Path $Root 'vendor'
$Wheels    = Join-Path $Vendor 'wheels'

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Write-Ok([string]$Text)   { Write-Host "    $Text" -ForegroundColor Green }
function Write-Info([string]$Text) { Write-Host "    $Text" -ForegroundColor DarkGray }
function Write-Warn2([string]$Text) { Write-Host "    $Text" -ForegroundColor Yellow }
function Fail([string]$Text) {
    Write-Host ''
    Write-Host "ERROR: $Text" -ForegroundColor Red
    exit 1
}

function Test-CommandExists([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# --------------------------------------------------------------------------- #
Write-Host ''
Write-Host '  OllaVoice - local speech for Ollama' -ForegroundColor White
Write-Host '  ===================================' -ForegroundColor DarkGray
Write-Host ''

# --------------------------------------------------------------------------- #
# 1. Python
# --------------------------------------------------------------------------- #
Write-Step 'Checking Python'

if (-not $PythonExe) {
    foreach ($Candidate in @('py', 'python', 'python3')) {
        if (Test-CommandExists $Candidate) { $PythonExe = $Candidate; break }
    }
}
if (-not $PythonExe) {
    Fail @"
Python was not found. Install Python 3.10 - 3.13 from https://www.python.org/downloads/windows/
(tick "Add python.exe to PATH"), then run this script again.
"@
}

# Resolve the version through the launcher when we were given "py".
$VersionArgs = @('-c', 'import sys; print("%d.%d" % sys.version_info[:2])')
if ($PythonExe -eq 'py') { $VersionArgs = @('-3') + $VersionArgs }
try {
    $VersionOutput = & $PythonExe @VersionArgs 2>$null
    $VersionExit = $LASTEXITCODE
    $PyVersion = (($VersionOutput | ForEach-Object { [string]$_ }) -join "`n").Trim()

    if ($VersionExit -ne 0) {
        $PyVersion = ''
    }
} catch {
    $PyVersion = ''
}
if (-not $PyVersion) { Fail "Could not determine the Python version from '$PythonExe'." }

$Major, $Minor = $PyVersion.Split('.') | ForEach-Object { [int]$_ }
if ($Major -ne 3 -or $Minor -lt 10 -or $Minor -gt 13) {
    Fail @"
Python $PyVersion is not supported. OllaVoice needs Python 3.10 - 3.13.
Install it from https://www.python.org/downloads/windows/ and re-run,
or pass -PythonExe "C:\Path\To\python.exe".
"@
}
Write-Ok "Python $PyVersion ($PythonExe)"

# --------------------------------------------------------------------------- #
# 2. Virtual environment
# --------------------------------------------------------------------------- #
Write-Step 'Creating the virtual environment'

if (Test-Path $Py) {
    Write-Ok ".venv already exists: $Venv"
} else {
    if ($PythonExe -eq 'py') { & py -3 -m venv $Venv } else { & $PythonExe -m venv $Venv }
    if ($LASTEXITCODE -ne 0) { Fail 'Could not create the virtual environment.' }
    Write-Ok "created $Venv"
}
if (-not (Test-Path $Py)) { Fail "venv python not found at $Py" }

& $Py -m pip install --upgrade pip --disable-pip-version-check --quiet
if ($LASTEXITCODE -ne 0) { Write-Warn2 'pip upgrade failed (continuing with the bundled pip)' }

# --------------------------------------------------------------------------- #
# 3. Packages
# --------------------------------------------------------------------------- #
Write-Step 'Installing Python packages'

$Packages = @('numpy', 'soundfile')
if (-not $NoStt) { $Packages += 'faster-whisper>=1.0' }
if (-not $NoTts) { $Packages += @('piper-tts>=1.3', 'kokoro-onnx>=0.6.1') }
if (-not $NoGui) { $Packages += 'PySide6>=6.6' }
$Packages += 'sounddevice>=0.4.6'      # fallback mic/playback for the terminal mode

$PipArgs = @('-m', 'pip', 'install', '--disable-pip-version-check')
if ($Offline -or (Test-Path $Wheels)) {
    if (-not (Test-Path $Wheels)) {
        Fail "Offline mode needs a wheels folder at $Wheels (see docs/INSTALL-windows.md)."
    }
    $PipArgs += @('--no-index', '--find-links', $Wheels)
    Write-Info "installing from $Wheels (no network)"
}
$PipArgs += $Packages
$PipArgs += @('-e', $Root)

& $Py @PipArgs
if ($LASTEXITCODE -ne 0) {
    Fail @"
pip install failed.
  - If you are behind a firewall, see docs/INSTALL-windows.md for the offline bundle route.
  - If a wheel is missing for Python $PyVersion, install Python 3.11 or 3.12 and re-run.
"@
}
Write-Ok "installed: $($Packages -join ', ')"

# --------------------------------------------------------------------------- #
# 4. Ollama
# --------------------------------------------------------------------------- #
Write-Step 'Checking Ollama'

$OllamaCmd = $null
foreach ($Candidate in @('ollama',
                         (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'),
                         'C:\Program Files\Ollama\ollama.exe')) {
    if (Test-CommandExists $Candidate) { $OllamaCmd = $Candidate; break }
    elseif (Test-Path $Candidate)      { $OllamaCmd = $Candidate; break }
}

if ($OllamaCmd) {
    Write-Ok "found ollama: $OllamaCmd"
    $Running = $false
    try {
        $Response = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 3 -UseBasicParsing
        if ($Response.StatusCode -eq 200) { $Running = $true }
    } catch { $Running = $false }

    if ($Running) {
        Write-Ok 'the Ollama server is running'
    } else {
        Write-Warn2 'the Ollama server is not running - trying to start it'
        Start-Process -FilePath $OllamaCmd -ArgumentList 'serve' -WindowStyle Hidden
        Start-Sleep -Seconds 4
    }
} else {
    Write-Warn2 @"
Ollama was not found. OllaVoice needs it for the language model.
  Install with:  winget install Ollama.Ollama
  Or download:   https://ollama.com/download/windows
You can finish this setup first; `ollavoice setup` can be re-run at any time.
"@
}

# --------------------------------------------------------------------------- #
# 5. Setup wizard (models)
# --------------------------------------------------------------------------- #
Write-Step 'Downloading speech models and verifying the loop'

$SetupArgs = @('-m', 'ollavoice', 'setup')
if ($Yes)        { $SetupArgs += '--yes' }
if ($Offline)    { $SetupArgs += '--offline' }
if ($Tier)       { $SetupArgs += @('--tier', $Tier) }
if ($Model)      { $SetupArgs += @('--model', $Model) }
if ($Language)   { $SetupArgs += @('--language', $Language) }
if ($TtsEngine)  { $SetupArgs += @('--tts-engine', $TtsEngine) }
if ($TtsVoice)   { $SetupArgs += @('--tts-voice', $TtsVoice) }
if ($SttModel)   { $SetupArgs += @('--stt-model', $SttModel) }
if ($Pull)       { $SetupArgs += '--pull' }
if ($NoPull)     { $SetupArgs += '--no-pull' }
if ($NoStt)      { $SetupArgs += '--skip-stt' }
if ($NoTts)      { $SetupArgs += '--skip-tts' }

& $Py @SetupArgs
$SetupExit = $LASTEXITCODE
if ($SetupExit -ne 0) {
    Write-Warn2 "setup finished with warnings (exit $SetupExit) - see the messages above"
}

# --------------------------------------------------------------------------- #
# 6. Shortcuts
# --------------------------------------------------------------------------- #
if (-not $NoShortcuts -and -not $NoGui) {
    Write-Step 'Creating Start Menu shortcuts'
    try {
        $StartMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\OllaVoice'
        New-Item -ItemType Directory -Force -Path $StartMenu | Out-Null

        $Shell = New-Object -ComObject WScript.Shell

        $Gui = $Shell.CreateShortcut((Join-Path $StartMenu 'OllaVoice.lnk'))
        $Gui.TargetPath = (Join-Path $Venv 'Scripts\pythonw.exe')
        $Gui.Arguments = '-m ollavoice gui'
        $Gui.WorkingDirectory = $Root
        $Gui.Description = 'Talk to your local Ollama models'
        $Gui.IconLocation = "$env:SystemRoot\System32\shell32.dll,12"
        $Gui.Save()

        $Term = $Shell.CreateShortcut((Join-Path $StartMenu 'OllaVoice (terminal).lnk'))
        $Term.TargetPath = (Join-Path $Venv 'Scripts\ollavoice.exe')
        $Term.Arguments = 'chat'
        $Term.WorkingDirectory = $Root
        $Term.Description = 'Voice chat in the terminal'
        $Term.Save()

        $Doctor = $Shell.CreateShortcut((Join-Path $StartMenu 'OllaVoice health check.lnk'))
        $Doctor.TargetPath = (Join-Path $Venv 'Scripts\ollavoice.exe')
        $Doctor.Arguments = 'doctor --test'
        $Doctor.WorkingDirectory = $Root
        $Doctor.Save()

        Write-Ok "shortcuts in $StartMenu"
    } catch {
        Write-Warn2 "could not create shortcuts: $($_.Exception.Message)"
    }
}

# --------------------------------------------------------------------------- #
# 7. Done
# --------------------------------------------------------------------------- #
Write-Host ''
Write-Host '  Done.' -ForegroundColor Green
Write-Host ''
Write-Host '  Start the desktop app:' -ForegroundColor White
Write-Host "      & '$Py' -m ollavoice gui" -ForegroundColor Cyan
Write-Host '  or use the Start Menu entry "OllaVoice".' -ForegroundColor DarkGray
Write-Host ''
Write-Host '  Talk in this terminal:' -ForegroundColor White
Write-Host "      & '$(Join-Path $Venv 'Scripts\ollavoice.exe')' chat" -ForegroundColor Cyan
Write-Host ''
Write-Host '  Useful commands:' -ForegroundColor White
Write-Host '      ollavoice doctor --test      re-run the self-test' -ForegroundColor DarkGray
Write-Host '      ollavoice config --tier high switch performance preset' -ForegroundColor DarkGray
Write-Host '      ollavoice voices --all       browse the voice catalogue' -ForegroundColor DarkGray
Write-Host ''
Write-Host '  Activate the venv in any shell with:' -ForegroundColor White
Write-Host "      & '$(Join-Path $Venv 'Scripts\Activate.ps1')'" -ForegroundColor Cyan
Write-Host ''

if ($SetupExit -ne 0) { exit $SetupExit }
exit 0
