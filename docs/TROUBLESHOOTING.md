# Troubleshooting

Start here:

```powershell
ollavoice doctor --test
```

That prints your hardware, whether Ollama is reachable, which speech engines are
available, what models are on disk, and then actually synthesises a sentence and
transcribes it back. Most problems identify themselves in that output.

Logs live in `%APPDATA%\OllaVoice\logs\` (`gui-YYYYMMDD.log`, and `crash.log` if a
packaged build failed to start).

---

## Contents

- [Install problems](#install-problems)
- [Ollama problems](#ollama-problems)
- [Microphone problems](#microphone-problems)
- [No sound / bad sound](#no-sound--bad-sound)
- [Transcription problems](#transcription-problems)
- [Speed problems](#speed-problems)
- [GUI problems](#gui-problems)
- [Hotkey problems](#hotkey-problems)
- [Offline and network problems](#offline-and-network-problems)
- [Still stuck](#still-stuck)

---

## Install problems

**`pip install` fails with "Could not find a version that satisfies faster-whisper"**

Your Python is too new or too old for the published wheels. Check
`python --version` — you need 3.10 to 3.13, and 3.11/3.12 have the best coverage.
Install 3.12 and point the installer at it:

```powershell
.\install-windows.bat -PythonExe "C:\Program Files\Python312\python.exe"
```

**`Activate.ps1 cannot be loaded because running scripts is disabled`**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Or skip activation and call `.venv\Scripts\ollavoice.exe` directly.

**`DLL load failed while importing _pywrap_ct2` / `onnxruntime_pybind11_state`**

Missing Visual C++ runtime. Install the
[VC++ 2015-2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
and retry. If it persists, reinstall the wheel:
`pip install --force-reinstall --no-cache-dir ctranslate2`.

**`OllaVoice crashed: illegal hardware instruction` (or instant exit, no message)**

CTRANSlate2 requires AVX2 on x86-64. `ollavoice doctor` reports `AVX2=no` if your
CPU lacks it (pre-2013 Intel, pre-2017 AMD). Options: use the `whisper-cpp`
engine with an SSE-only build, use the built-in `windows` engine, or run on a
newer machine.

**Antivirus deleted something / the app vanishes on launch**

`ctranslate2` and `onnxruntime` load large unsigned native DLLs. Check your AV
quarantine and whitelist `.venv\Lib\site-packages`.

**`piper-tts` or `kokoro-onnx` missing but you want them later**

```powershell
pip install piper-tts kokoro-onnx
ollavoice config set tts.engine kokoro
```

---

## Ollama problems

**"Cannot reach Ollama at http://127.0.0.1:11434"**

Ollama is not running.

```powershell
ollama serve                  # in a terminal, or
start ollama                  # the tray app
```

If you installed it with `winget install Ollama.Ollama`, it normally starts with
Windows and puts an icon in the tray. If you bound Ollama to a different address,
tell OllaVoice: `ollavoice config set ollama.host http://127.0.0.1:11435`.

**"Ollama does not have 'qwen3:4b' yet"**

```powershell
ollama list                   # what you actually have
ollama pull qwen3:4b          # or pick one you already have:
ollavoice config set ollama.model llama3.2:3b
```

You can also choose from the dropdown in the toolbar — it is populated from
`ollama list`.

**The first answer takes 20+ seconds, later ones are fast**

That is the model loading into RAM/VRAM. `ollama ps` shows what is resident.
OllaVoice sends `keep_alive: 10m` so it stays warm between turns; raise it with
`ollavoice config set ollama.keep_alive 30m` if you have the memory to spare.

**Answers are walls of markdown that the voice reads badly**

The default system prompt already asks for plain prose. Strengthen it in
Settings → General, or:

```powershell
ollavoice config set ollama.system_prompt "Answer in one or two short spoken sentences. No markdown, no lists."
```

Also confirm `tts.strip_markdown` is on (it is by default) — that removes code
fences, headings and links before synthesis regardless of what the model does.

**Out of memory / Ollama kills the model mid-answer**

Drop to a smaller model or tier: `ollavoice config --tier low`, and reduce
context with `ollavoice config set ollama.num_ctx 2048`. Running an 8B model,
Whisper and a TTS voice at once needs real headroom.

---

## Microphone problems

**The level meter does not move when I speak**

1. Windows Settings → Privacy & security → Microphone → *Let desktop apps access
   your microphone* = **On**. This is the most common cause by far.
2. Settings → Audio devices → **Rescan devices**, then pick the right microphone.
   Windows frequently defaults to a webcam or a monitor's mic array.
3. Windows Sound settings → your microphone → Levels: make sure it is not muted
   and the volume is above 50.

**"recorded 1.4 s of audio" but the transcript is empty**

You are recording silence — see above. It can also happen if the mic works but is
far away: watch the meter and aim for peaks around −20 to −6 dB.

**Audio is clipping (meter goes red, transcript has wrong words)**

Turn the microphone level down in Windows Sound settings. The meter turns red
above −3 dBFS.

**It transcribes its own answer**

`audio.mute_mic_while_speaking` is on by default; if you turned it off, turn it
back on. With speakers (not headphones) in a small room, some bleed is
unavoidable — hands-free mode is more sensitive to it than push-to-talk, because
push-to-talk only listens while you hold the key.

**`No usable audio backend`**

```powershell
pip install PySide6         # recommended: mic + playback, no system deps
pip install sounddevice     # alternative: PortAudio, bundled in the Windows wheel
```

On Linux you may instead need `libportaudio2`, or `ffmpeg` / `alsa-utils` for the
`system` backend.

**The wrong sample rate is negotiated**

Some devices refuse 16 kHz. OllaVoice asks for 16 kHz mono int16 and falls back to
the device's preferred format, resampling afterwards — you should not need to do
anything. If a specific headset misbehaves, try
`ollavoice config set audio.sample_rate 44100`.

---

## No sound / bad sound

**No audio at all, but the text answer appears**

1. Is *Speak replies* ticked (bottom bar) and `tts.engine` not `none`?
   `ollavoice config get tts.engine`
2. Settings → Audio devices → pick the right output device.
3. Test the voice directly: Settings → Voice output → **Test voice**, or
   `ollavoice speak "testing one two three"`.
4. Check Windows Volume Mixer — the app may be muted individually.

**"Piper voice 'X' is not installed"**

```powershell
ollavoice voices                                   # what you have
ollavoice voices --install en_US-lessac-high       # get one
ollavoice setup --tts-voice en_US-lessac-high
```

**"Kokoro model not installed"**

```powershell
ollavoice setup --tts-engine kokoro
```

That fetches `kokoro-v1.0.int8.onnx` (114 MB) + `voices-v1.0.bin` (28 MB).

**Kokoro is too slow**

Expected: it is roughly 4× slower than Piper on CPU. Use `kokoro-v1.0.int8.onnx`
(the default), or switch to Piper for conversation and keep Kokoro for
`ollavoice speak -f document.md`.

**The voice mispronounces things / says "hash hash hash"**

Make sure `tts.strip_markdown` is on. For stubborn cases, switch engine (Piper
and Kokoro phonemise differently), or try a different voice — quality varies a
lot between them: `ollavoice voices --all --language en`.

**The voice reads URLs and email addresses**

It should not: `strip_for_speech()` removes them. If one gets through, tell me the
exact text and I will add the pattern.

**Robotic voice**

You are on the `sapi` engine (the built-in Windows voices). Switch to a neural
one: `ollavoice config set tts.engine piper`.

**Speech starts before the answer is finished — is that a bug?**

No, that is the point. Sentence-level streaming means the first sentence is
spoken while the model is still generating. Turn it off with
`ollavoice config set tts.stream_sentences false` if you would rather wait.

---

## Transcription problems

**Whisper hallucinates ("Thanks for watching!", "[Music]", repeated words)**

* `stt.vad_filter` on (default) — filters non-speech
* `stt.condition_on_previous_text` off (default) — stops repetition loops
* Use a bigger model: `ollavoice config set stt.model small.en`
* Record less silence: trim by using push-to-talk rather than hands-free

**It hears "Obama" instead of "Ollama"**

Add your vocabulary to the prompt bias:

```powershell
ollavoice config set stt.initial_prompt "Ollama, OllaVoice, GGUF, llama.cpp, Kubernetes."
```

Whisper weights proper nouns it has been told to expect. This is the single
biggest accuracy win for technical dictation.

**Wrong language detected**

Pin it: `ollavoice config set stt.language en`. Note that `auto` forces the
multilingual model (`base` instead of `base.en`), which is slightly worse at
English than the English-only checkpoint.

**Accuracy is poor generally**

Go up a size. On the `low` tier you have `base.en`; `small.en` is markedly better
and still faster than realtime on a modern CPU:

```powershell
ollavoice config set stt.model small.en
ollavoice config set stt.beam_size 5
```

**Long recordings lose the middle**

Whisper works in 30 s windows. `ollavoice listen` splits on silence automatically
for anything over 25 s; the GUI sends one clip per turn, so prefer shorter turns
for long dictation.

---

## Speed problems

**Everything is slow**

Run `ollavoice doctor` and look at the numbers. Then, in rough order of payoff:

1. `ollavoice config --tier low` — smaller Whisper, smaller LLM
2. `ollavoice config set stt.model base.en` and `stt.beam_size 1`
3. `ollavoice config set tts.engine piper` (Kokoro is ~4× slower on CPU)
4. `ollavoice config set ollama.num_ctx 2048`
5. Check `stt.device`: if you have an NVIDIA GPU it should say `cuda`, not `cpu`.
   If it says `cpu` and you have a GPU, CUDA/cuDNN is not installed for
   CTranslate2 — see below.

**GPU is not being used for Whisper**

`ollavoice doctor` prints what Ollama sees. For faster-whisper you need the CUDA
12 + cuDNN 9 runtime DLLs on PATH (or `pip install nvidia-cublas-cu12
nvidia-cudnn-cu12`). Then:

```powershell
ollavoice config set stt.device cuda
ollavoice config set stt.compute_type float16
```

If the CUDA libraries are missing, OllaVoice quietly falls back to CPU rather
than failing — check the `stt:` field in the status bar.

**First run is much slower than later ones**

Model loading. The GUI warms both engines on a worker thread at startup; the
first *turn* also pays the Ollama model load.

**Latency breakdown**

`ollavoice doctor --test` and the timing line under each answer show
`whisper Xs · first token Ys · answer Zs`. That tells you which stage to attack.

---

## GUI problems

**`ollavoice gui` says PySide6 is missing**

```powershell
pip install PySide6
```

**PySide6 is installed but Qt will not load**

The message names the missing library. On Linux:

```bash
sudo apt install libglib2.0-0 libgl1 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1
```

On Windows: `pip install --force-reinstall PySide6`.

**The window is off-screen / the wrong size**

```powershell
ollavoice gui --reset-window
```

**I closed the window and it did not quit**

By design — it minimises to the tray so your push-to-talk hotkey keeps working.
Use the tray icon → Quit, or turn it off with
`ollavoice config set gui.minimize_to_tray false`.

**Two instances are running**

The single-instance guard uses a PID lock file at
`%APPDATA%\OllaVoice\gui.lock`. If a previous crash left a stale lock and the app
refuses to start, delete that file.

**The transcript view is sluggish on very long conversations**

Ctrl+N starts a new conversation. History sent to the model is capped separately
by `ollama.history_turns`.

---

## Hotkey problems

**Push-to-talk does not work when OllaVoice is in the background**

The status bar shows either `hotkeys: push_to_talk, stop_speaking` (global, via
`RegisterHotKey`) or `... (focus needed)` (Qt shortcut fallback). If you see the
latter, another application already owns that combination — Discord, Teams,
GeForce Experience and Steam grab a lot of them. Change yours in Settings →
Audio devices → Hotkeys, e.g. `Ctrl+Shift+F9`.

**The hotkey works but I cannot type spaces in the text box**

Space only triggers recording when the text box does *not* have focus. If it
still does, use the global hotkey for push-to-talk and leave Space alone.

**Hotkeys stopped working after sleep/resume**

Re-register them: Settings → Audio devices → toggle *Enable global hotkeys*, or
restart the app.

---

## Offline and network problems

**Setup hangs on download**

Something is blocking Hugging Face or GitHub. Either set a mirror:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
ollavoice setup
```

or use the offline bundle route in [INSTALL-windows.md](INSTALL-windows.md#5-offline--air-gapped-install).

**`OLLAVOICE_OFFLINE is set but X is missing`**

That is deliberate: in offline mode a missing file is an immediate, specific
error instead of a hang. The message names the path to copy the file into.

**Does it ever phone home?**

No. The only outbound connections are the ones `ollavoice setup` makes to
download models, and `ollama pull`. At runtime the process talks to
`127.0.0.1:11434` and nothing else. There is no telemetry, no update check and no
crash reporting in the codebase. Once a Whisper model is cached,
`HF_HUB_OFFLINE=1` is set automatically so even the revision check is skipped.
Test it: disable your network adapter and run `ollavoice doctor --test`.

**Downloads are corrupt**

Kokoro and Piper downloads are verified where checksums are published, and all
downloads write to a `.part` file and are moved into place atomically, so an
interrupted download cannot leave a half-written model. Delete the file and
re-run setup; a `.part` file resumes rather than restarting.

---

## Still stuck

Collect this and file an issue:

```powershell
ollavoice doctor --json > doctor.json
ollavoice config show   > config.json
```

plus the tail of `%APPDATA%\OllaVoice\logs\gui-YYYYMMDD.log`. Redact anything you
consider private — the config includes your system prompt and your Ollama host,
and the transcript folder (if enabled) contains your conversations.
