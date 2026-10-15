# Installing OllaVoice on Windows

Contents

1. [Requirements](#1-requirements)
2. [Automatic install](#2-automatic-install-recommended)
3. [Manual install](#3-manual-install)
4. [First run](#4-first-run)
5. [Offline / air-gapped install](#5-offline--air-gapped-install)
6. [Building a standalone OllaVoice.exe](#6-building-a-standalone-ollavoiceexe)
7. [Windows-specific notes](#7-windows-specific-notes)
8. [Upgrading and uninstalling](#8-upgrading-and-uninstalling)

---

## 1. Requirements

| | |
|---|---|
| OS | Windows 10 (1903+) or Windows 11, 64-bit. ARM64 works for the Python route |
| Ollama | [OllamaSetup.exe](https://ollama.com/download/windows), plus one pulled model |
| Python | 3.10 – 3.13, 64-bit. **3.11 or 3.12 is the safest choice** — every wheel we need is published for those |
| Disk | ~1 GB for the balanced tier (speech models) + your Ollama models |
| Audio | Any microphone and output device |
| GPU | Optional. An NVIDIA GPU with the CUDA runtime makes Whisper noticeably faster; nothing else needs one |

Check what you have:

```powershell
python --version          # or: py -3 --version
ollama --version
ollama list
```

If `python` is not recognised, install it from
[python.org](https://www.python.org/downloads/windows/) and **tick "Add
python.exe to PATH"** in the first screen of the installer.

If your Python is 3.14 or newer, some wheels may not exist yet. Install 3.12
alongside it and point the installer at it: `install-windows.bat -PythonExe
"C:\Program Files\Python312\python.exe"`.

---

## 2. Automatic install (recommended)

```powershell
cd C:\path\to\ollavoice
.\install-windows.bat
```

Or, if double-clicking `.bat` files is blocked by policy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Useful flags (all optional):

```powershell
.\install-windows.bat -Tier high                  # skip hardware auto-detection
.\install-windows.bat -Model llama3.1:8b          # which Ollama model to use/pull
.\install-windows.bat -Language de                # German instead of English
.\install-windows.bat -TtsEngine kokoro           # start with the nicer voice
.\install-windows.bat -TtsVoice en_GB-alan-medium # specific Piper voice
.\install-windows.bat -NoGui                      # terminal-only, saves ~600 MB
.\install-windows.bat -NoPull                     # configure, but don't download an LLM
.\install-windows.bat -Pull                       # do download the LLM now
.\install-windows.bat -Yes                        # non-interactive
```

What the script does, in order:

1. Finds Python and checks the version is supported
2. Creates `.venv` next to the script (nothing touches your system Python)
3. `pip install`s the runtime packages and OllaVoice itself
4. Looks for `ollama.exe`, starts `ollama serve` if it is not running
5. Runs `ollavoice setup`: detects CPU/RAM/VRAM, picks a tier, downloads the
   Whisper model and voice(s), then **verifies the loop** by synthesising a
   sentence and transcribing it back
6. Creates Start Menu shortcuts under *OllaVoice*

Re-running it is safe — every step skips work that is already done.

---

## 3. Manual install

```powershell
cd C:\path\to\ollavoice

# 1. virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. packages
pip install --upgrade pip
pip install -r requirements-windows.txt      # or: pip install -e ".[all]"

# 3. models + self-test
ollavoice setup

# 4. go
ollavoice gui
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Or skip activation entirely and call the venv binaries directly:

```powershell
.venv\Scripts\ollavoice.exe gui
```

### Installing the pieces separately

The extras are independent, so you can install only what you need:

```powershell
pip install -e ".[gui]"     # PySide6 desktop app
pip install -e ".[stt]"     # faster-whisper
pip install -e ".[tts]"     # piper + kokoro
pip install -e ".[audio]"   # sounddevice (mic fallback for terminal mode)
```

---

## 4. First run

```powershell
ollavoice doctor --test
```

You should see hardware details, the Ollama status, both speech engines marked
`ok`, and a self-test that synthesises a sentence and transcribes it back. If any
line says `fail`, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Then launch the app:

```powershell
ollavoice gui
```

**First launch is slower than later ones** — Whisper and the voice load their
weights (a few seconds). Hold **Space**, speak, release. Your words appear as a
transcript, the answer streams in, and the voice starts talking before the model
has finished.

Two things worth setting straight away:

* **Settings → Voice input → Hands-free** if you would rather it stop listening
  automatically than hold a key.
* **Settings → Audio devices → Rescan devices** if it picked the wrong
  microphone (Windows often defaults to a webcam mic).

### Firewall

OllaVoice never listens on a port unless you enable the local speech API
(Settings → Advanced, or `ollavoice serve`). If you do enable it, Windows may
show a firewall prompt — allow it on **private networks only**, or decline and
keep using `127.0.0.1`, which needs no firewall rule at all.

---

## 5. Offline / air-gapped install

Two-machine workflow: a connected PC prepares a bundle, the offline PC consumes it.

### On the connected PC

```powershell
# 1. Python wheels for the target platform
mkdir vendor\wheels
pip download -r requirements-windows.txt -d vendor\wheels
pip download . -d vendor\wheels --no-deps

# 2. Install normally so the models are downloaded...
.\install-windows.bat

# 3. ...then copy them into a portable bundle
.venv\Scripts\ollavoice.exe bundle vendor\models --wheels vendor\wheels
```

Zip the whole project folder (or at least `vendor\`) and copy it across.

### On the offline PC

```powershell
cd C:\ollavoice
.\install-windows.bat -Offline
```

`-Offline` sets `OLLAVOICE_OFFLINE=1`, which makes pip use
`--no-index --find-links vendor\wheels` and makes every model download read from
`vendor\models` instead of the network. If a file is missing you get an error
naming the exact path to copy it to, rather than a hang.

The bundle layout:

```
vendor\
├── wheels\                  every .whl needed, for the target Python version
└── models\
    ├── whisper\             faster-whisper cache (models--Systran--faster-whisper-*)
    ├── piper\               <voice>.onnx + <voice>.onnx.json
    └── kokoro\              kokoro-v1.0.int8.onnx + voices-v1.0.bin
```

> **Match the Python version** between the two machines, or `pip download` will
> fetch wheels the target cannot use. `pip download --python-version 3.12
> --only-binary=:all:` works if you must prepare on a different version.

### Model URLs, if you prefer to download by hand

| Artifact | URL |
|---|---|
| Whisper | `https://huggingface.co/Systran/faster-whisper-<size>` (or let `faster-whisper` fetch it) |
| Piper voice | `https://huggingface.co/rhasspy/piper-voices/resolve/main/<fam>/<lang_REGION>/<name>/<quality>/<lang_REGION>-<name>-<quality>.onnx` (+ the same with `.onnx.json`) |
| Kokoro | `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.int8.onnx` and `voices-v1.0.bin` |
| Ollama model | `ollama pull <model>` on the connected machine, then copy `%USERPROFILE%\.ollama\models` |

Behind a firewall with an HF mirror:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
ollavoice setup
```

---

## 6. Building a standalone OllaVoice.exe

PyInstaller cannot cross-compile, so **build on Windows**:

```powershell
.\scripts\build-windows-exe.ps1              # dist\OllaVoice\  (folder, fast start)
.\scripts\build-windows-exe.ps1 -OneFile     # dist\OllaVoice.exe (single file, slow start)
```

Expect ~1.5 – 2 GB. That size is `ctranslate2`, `onnxruntime`, PySide6/Qt and two
copies of the espeak-ng phoneme data (one from `piper`, one from
`espeakng_loader`); the spec excludes tkinter, matplotlib, scipy, pandas and
torch to keep it from growing further. Prefer the folder build — a onefile
executable has to unpack all of that to a temp directory on every launch.

The exe does **not** contain model weights. It reads them from
`%LOCALAPPDATA%\OllaVoice\models`, so ship it with either:

* a first run on a machine with internet (the setup wizard fetches them), or
* an offline bundle copied into that folder, or
* `OLLAVOICE_HOME` set to a directory on a USB stick for a portable install.

The build script runs the test suite first; pass `-SkipTests` to skip it.
`.github/workflows/tests.yml` builds the exe automatically on version tags.

---

## 7. Windows-specific notes

**Microphone permission.** Settings → Privacy & security → Microphone →
*Let desktop apps access your microphone* must be **On**. This is the single most
common cause of "recorded 1.4 s of audio" and an empty transcript.

**Wrong microphone.** Windows defaults to whatever was plugged in last. Fix it
in Settings → Audio devices, or system-wide in Sound settings. The level meter in
OllaVoice should move when you speak; if it does not, the wrong device is
selected.

**Hearing itself.** If the assistant transcribes its own voice, make sure
`audio.mute_mic_while_speaking` is on (it is by default) and that your speakers
are not feeding straight into the microphone. Headphones remove the problem
entirely.

**Global hotkeys.** Push-to-talk uses `RegisterHotKey`, so it works when
OllaVoice is unfocused. If another app already owns your combination
(Discord, Teams and some GPU utilities grab a lot of them), registration fails
silently for that one key and the status bar shows *hotkeys: ... (focus needed)*.
Pick a different combination in Settings → Audio devices → Hotkeys.

**Antivirus.** `ctranslate2` and `onnxruntime` load large unsigned native DLLs,
which some AV products flag. If OllaVoice vanishes on startup, check your AV
quarantine and whitelist the `.venv\Lib\site-packages` folder.

**Visual C++ runtime.** Needed by the native wheels. If you see
`DLL load failed while importing ...`, install the
[VC++ 2015-2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

**Paths with spaces or accents** are fine; everything is handled as Unicode.

**Long paths.** If you extract the project somewhere very deep, enable long path
support (`gpedit` → *Enable Win32 long paths*) — the espeak-ng data tree has some
deeply nested filenames.

---

## 8. Upgrading and uninstalling

**Upgrade:**

```powershell
cd C:\path\to\ollavoice
git pull                                  # or re-extract the new release
.\install-windows.bat                     # re-runs safely, skips what exists
ollavoice doctor --test
```

Config and models are preserved; they live outside the project folder.

**Uninstall:**

```powershell
Remove-Item -Recurse -Force C:\path\to\ollavoice          # code + .venv
Remove-Item -Recurse -Force "$env:APPDATA\OllaVoice"      # config, logs, transcripts
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\OllaVoice" # models (~1 GB)
Remove-Item -Recurse -Force "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\OllaVoice"
```

Ollama and its models are separate and untouched by this.

**Where everything lives** — `ollavoice config path` prints the config file, and
Settings → Advanced (or `ollavoice doctor`) lists all of them:

```
%APPDATA%\OllaVoice\config.json          settings
%APPDATA%\OllaVoice\logs\                logs (gui-YYYYMMDD.log, crash.log)
%APPDATA%\OllaVoice\transcripts\         opt-in chat history, JSONL by day
%LOCALAPPDATA%\OllaVoice\models\         whisper\  piper\  kokoro\
%LOCALAPPDATA%\OllaVoice\audio\          scratch WAV files (auto-deleted)
```

Set `OLLAVOICE_HOME` to move config *and* models into one directory — handy for
a portable install on a USB drive or a non-system disk.
