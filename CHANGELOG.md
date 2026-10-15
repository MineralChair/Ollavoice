# Changelog

## 1.0.0

First release. Adds the two things Ollama does not ship natively — speech input
and speech output — as a companion app rather than a fork, so it keeps working as
Ollama evolves.

**Speech input**

- OpenAI Whisper via `faster-whisper` (int8 CPU / float16 CUDA, chosen from
  detected hardware)
- Optional `whisper.cpp` backend for an existing local build
- Optional Ollama backend that probes `/api/transcriptions` and explains itself if
  the installed server has no such endpoint
- Optional Windows built-in recogniser via `System.Speech` (no model download)
- Prompt biasing for proper nouns, VAD filtering, and cleanup of Whisper's usual
  silence hallucinations

**Speech output**

- Piper (default, ~60 MB/voice, streams chunk-by-chunk)
- Kokoro-82M through `kokoro-onnx` (int8, more natural)
- Windows SAPI voices through PowerShell (zero download)
- Silent mode
- Markdown/HTML stripped and symbols expanded before synthesis, so code fences
  and `%`/`&`/`→` are not read aloud literally

**Pipeline**

- Sentence-level streaming: the reply is spoken while the model is still
  generating
- A dedicated speech queue thread, so a voice slower than realtime (Kokoro on a
  small CPU) cannot stall token streaming
- Adaptive VAD with a calibration window, hysteresis and hangover for hands-free
  listening
- Push-to-talk with pre-roll, so the first phoneme is not clipped
- Microphone muted during playback to stop the assistant transcribing itself
- Per-turn timings (whisper / first token / answer / first audio)

**Interfaces**

- Native Qt 6 desktop app: dark theme, live level meter, streaming transcript,
  model and voice pickers, system tray, five-tab settings dialog
- Global hotkeys via `RegisterHotKey` (work when the app is unfocused), with an
  in-app shortcut fallback
- Terminal REPL (`ollavoice chat`) with slash commands
- Local HTTP API with OpenAI-compatible `/v1/audio/transcriptions` and
  `/v1/audio/speech`, plus `/api/chat` for speech-in/speech-out
- Dependency-free browser UI served by that API

**Setup and operations**

- Hardware detection (CPU, AVX2, RAM, VRAM via `nvidia-smi`/WMI/Ollama) with
  automatic `low`/`balanced`/`high` tier selection
- `ollavoice doctor --test` synthesises a sentence and transcribes it back to
  prove the loop works
- Resumable, checksummed downloads with mirror support
- Offline installs: `ollavoice bundle` + `setup --offline`, and `HF_HUB_OFFLINE`
  set automatically once a Whisper model is cached so nothing phones home later
- Windows installer (`install-windows.bat` / `scripts\setup.ps1`) creating a venv,
  starting Ollama if needed, and adding Start Menu shortcuts
- PyInstaller spec and build script for a standalone `OllaVoice.exe`

**Testing**

- 260+ tests: pure logic, a fake Ollama HTTP server driving the real pipeline,
  and integration tests that run actual Piper and Whisper and assert the words
  survive the round trip
- GUI tests run against a permissive fake `PySide6` when Qt cannot load, so they
  work in headless CI
