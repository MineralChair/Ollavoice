# OllaVoice architecture

Why this exists: Ollama is a text-in/text-out runtime. It has no microphone path
and no audio output path — the
[native TTS request](https://github.com/ollama/ollama/issues/11021) is still open
and the [API reference](https://docs.ollama.com/api) lists `generate`, `chat`,
`embed`, `ps`, `tags`, `pull`, `create`, `copy`, `delete` and cloud endpoints, but
nothing for audio. So OllaVoice does not patch Ollama. It sits beside it and adds
the two missing ends of the pipeline, using the same local-server model Ollama
itself established.

```
                        ┌──────────────────────────────────────────────┐
                        │                OllaVoice process             │
                        │                                              │
 microphone ──▶ audioio ─┤ vad.py (optional) ──▶ stt_engines ──┐        │
                        │                     (faster-whisper) │       │
 keyboard ──────────────┤──────────────────────────────────────┤       │
                        │                                      ▼       │
 browser (MediaRecorder) ┤──▶ server.py ──────────────▶ pipeline.py     │
                        │                              │       │       │
                        │                              ▼       ▼       │
                        │                     ollama_client  tts_engines
                        │                              │       │       │
                        └──────────────────────────────┼───────┼───────┘
                                                       ▼       ▼
                                          http://127.0.0.1:11434   audioio ──▶ speakers
                                             (Ollama, separate)
```

Everything in the box is one Python process. Ollama is a separate local server
that OllaVoice talks to over loopback HTTP, exactly like `ollama run` does.

---

## Module map

| Module | Responsibility | Depends on |
|---|---|---|
| `wav.py` | `AudioClip` (mono int16 PCM + rate), WAV I/O, resampling, dBFS maths | numpy |
| `vad.py` | Streaming `SpeechGate` (calibrated noise floor, hangover), offline `trim_clip` / `split_on_silence` | numpy |
| `textsplit.py` | Markdown → spoken prose; `SentenceSplitter` for token streams | – |
| `stt_engines.py` | `SttEngine` ABC + faster-whisper / whisper.cpp / Ollama / Windows | faster-whisper |
| `tts_engines.py` | `TtsEngine` ABC + Piper / Kokoro / SAPI / silent | piper, kokoro-onnx |
| `audioio.py` | Recorder + Player with three interchangeable backends (Qt / sounddevice / system tools) | PySide6 *or* sounddevice |
| `ollama_client.py` | Dependency-free HTTP client for `/api/chat`, `/api/tags`, `/api/ps`, `/api/pull` | stdlib |
| `pipeline.py` | Orchestration: listen → transcribe → stream → speak, with events | all of the above |
| `config.py` | Typed config, tiers, JSON persistence with write-side validation | – |
| `hardware.py` | CPU/RAM/VRAM/AVX2 detection, tier recommendation, Whisper device choice | stdlib |
| `paths.py` | Where everything lives on disk | stdlib |
| `downloads.py` | Model downloads: resume, checksums, mirrors, offline mode, bundles | stdlib |
| `voices.py` | Piper catalogue access (swallowing the upstream CLI's stdout) | piper |
| `setup.py` | First-run wizard + verifier | all |
| `cli.py` / `repl.py` | Command line entry points | pipeline |
| `server.py` | Local HTTP API + browser UI | pipeline |
| `gui/` | Qt main window, settings dialog, worker threads, hotkeys, theme | PySide6 |

The dependency arrows only point one way: `gui`, `cli`, `repl` and `server` all
depend on `pipeline`; `pipeline` depends on the engines; nothing in the engine
layer knows a GUI exists. That is what lets the same pipeline drive four
different front-ends and be tested without audio hardware.

---

## The turn

`VoicePipeline.run_turn()` is the whole product in one function:

1. **Input.** Either an `AudioClip` from the microphone or text from a keyboard.
   If audio: `stt.transcribe()` → `Transcription`. Either way a single
   `transcript` event is emitted, so every front-end has exactly one path for
   "the user said this".
2. **History.** `ChatHistory.add_user()`; a sliding window (`ollama.history_turns`)
   keeps the context bounded. Nothing is persisted unless you enable transcripts.
3. **Generation.** `ollama_client.chat(..., stream=True)` yields NDJSON events.
   Each delta is re-emitted as a `token` event for live display.
4. **Sentence splitting.** Deltas also feed `SentenceSplitter`. As soon as a
   complete sentence exists it is handed to TTS — *while generation continues*.
5. **Synthesis, on its own thread.** `enqueue_speech(sentence)` pushes onto a
   queue drained by the `ollavoice-speech` worker, which synthesises and plays.
   Without that indirection a voice slower than realtime (Kokoro on a small CPU)
   stalls the token loop: the model would have to wait for the previous sentence
   to be synthesised *and* played before it was read again.
6. **Playback.** The microphone is muted for the duration when
   `audio.mute_mic_while_speaking` is set, so the assistant cannot hear itself.
7. **Completion.** A single `done` event carries `data={"final": True}`. The GUI
   unlocks on that, never on an intermediate event. `wait_for_playback()` is what
   a front-end calls when it wants to block until the voice has caught up.

Step 4 is the design decision that matters most. Speaking the whole answer after
generation finishes costs you the full generation time in silence; speaking each
sentence as it completes means audio starts after roughly one sentence of
inference. With `min_speak_chars = 24` the first chunk is usually spoken before
the model has produced its second sentence.

### Events

The pipeline never touches a widget. It calls `event_fn(Event(kind, text, phase,
data))` with these kinds:

| kind | payload |
|---|---|
| `phase` | `idle` / `listening` / `transcribing` / `thinking` / `speaking` / `error` |
| `level` | `data.db` (dBFS), `data.state` — drives the level meter |
| `transcript` | `text` + language, timings, engine, model |
| `token` | one streamed delta of the answer |
| `sentence` | a sentence was queued for speech |
| `audio` | a clip was synthesised, with `duration_s` / `synth_s` |
| `stats` / `done` / `info` / `error` | timings, completion, log lines, failures |

`emit()` swallows exceptions from the listener on purpose: a broken UI callback
must never be able to kill a turn.

---

## Threading

**GUI process**

```
main thread (Qt event loop)      worker thread (QThread)      speech thread
─────────────────────────       ───────────────────────      ─────────────────────
QAudioSource → readyRead         PipelineWorker.run_turn()    _speech_loop()
  → int16 chunks in a list         → pipeline.transcribe()      → queue.get()
QTimer(120 ms) → SpeechGate          (Whisper)                  → tts.synthesize_stream()
level meter, buttons, HTML         → ollama.chat(stream=True)   → player.play(blocking)
                                   → SentenceSplitter
                                   → enqueue_speech(sentence) ─────────▶
       ▲                           → wait_for_speech()
       │  Qt signals (queued)
       └──────────────────────────  Signals.token/done/error/...
```

Three threads, three jobs:

* **Main.** Microphone capture stays here because `QAudioSource` belongs to the
  thread that created it. It only appends numpy chunks; the VAD runs on a 120 ms
  `QTimer`. All widget updates happen here.
* **Worker.** Whisper, the Ollama stream and the sentence splitter.
* **Speech.** Synthesis and playback, draining a `queue.Queue`. Keeping this
  separate is what stops a slow voice from throttling generation.

GUI → worker calls go through `request_*` signals and worker → GUI through the
`Signals` bundle; both directions are queued connections, so no widget is ever
touched off the main thread and the UI needs no locks. ONNX Runtime sessions are
not concurrency-safe, so every synthesis is serialised behind `VoicePipeline._tts_lock`
(the speech thread and any `/v1/audio/speech` request share it).

Playback is non-blocking from the worker's point of view and the worker calls
`wait_for_playback()` at the end of a turn. That is what makes **Stop** able to
interrupt mid-sentence instead of being frozen until the whole answer has been
read out: `cancel()` sets the flag, drains the queue and stops the sink.

`RefreshWorker` polls Ollama every 20 s for the model list and engine
availability, also off-thread.

**HTTP server** uses `ThreadingHTTPServer` with a single `threading.Lock` around
a turn, because the speech engines are effectively single-threaded. The lock is
released *before* the response is written; otherwise a client that fires the next
request the instant it sees the last byte would race the unlock and get a
spurious 429 (this was a real bug, and there is a test for it).

---

## Audio backends

Three interchangeable implementations behind `make_recorder()` / `make_player()`:

| Backend | Record | Play | Notes |
|---|---|---|---|
| `qt` | `QAudioSource` | `QAudioSink` + `QBuffer` | Used by the GUI. No native audio dependency beyond Qt itself |
| `sounddevice` | `RawInputStream` + callback | `sd.play` | Terminal/server mode. Windows wheel bundles PortAudio |
| `system` | `ffmpeg` / `arecord` | `ffplay` / `aplay` / `SoundPlayer` | Last resort, needs external tools |

`choose_backend("auto")` picks the first that can actually load — note that
"PySide6 is installed" is not the same as "Qt loads": on Linux a missing
`libglib`/`libGL` makes the import raise `OSError`, and `qt_available()` catches
that so the CLI degrades to sounddevice instead of crashing.

Capture format is negotiated per device: we ask for 16 kHz mono int16 and, if the
device refuses, fall back to its preferred format at the same sample format and
resample in `AudioClip.resample()`. Everything downstream assumes mono int16, so
resampling happens once, at the edge.

---

## Speech-to-text

Default is **faster-whisper** (Whisper on CTranslate2):

* int8 on CPU, float16 on CUDA — chosen from detected hardware, overridable
* `cpu_threads = min(8, cores - 1)`: leaves a core for the GUI and Ollama, and
  CT2 scales poorly past ~8 anyway
* `vad_filter=True` and `condition_on_previous_text=False` by default, which
  between them remove most Whisper hallucination on silence
* `initial_prompt` biases proper nouns ("Ollama, Piper, Kokoro, Whisper") —
  without it, `tiny`/`base` reliably hear "Obama" instead of "Ollama"
* model choice follows language: `base.en` for English, `base` when the language
  is `auto` or anything else, because an English-only checkpoint cannot detect
  language at all

Alternatives, all optional:

* **whisper.cpp** — for people who already run llama.cpp. Drop `whisper-cli.exe`
  and a `ggml-*.bin` into `%LOCALAPPDATA%\OllaVoice\models\whisper.cpp` (or set
  `OLLAVOICE_WHISPERCPP_BIN` / `..._MODEL`) and it shells out to them. No Python
  ML stack involved at inference time.
* **Ollama** — probes `POST /api/transcriptions` at runtime. Current Ollama
  releases do not implement it, so the engine raises an error that says exactly
  that and points you at faster-whisper. It is here so that the day Ollama ships
  audio endpoints, it is a one-line config change.
* **Windows** — the recogniser built into Windows, driven through
  `System.Speech` and PowerShell. No model download at all; lowest accuracy.

Encoded uploads (webm/ogg/mp3) are decoded by the ffmpeg libraries bundled in
`faster-whisper`'s `av` dependency, which is why the browser UI needs no ffmpeg
on PATH.

---

## Text-to-speech

| Engine | Size | Speed (2-core CPU) | Notes |
|---|---|---|---|
| **Piper** | ~60 MB/voice | ~1.2× realtime | Default. Streams chunk-by-chunk. 100+ voices, ~25 languages |
| **Kokoro-82M** | ~140 MB | ~4× realtime | More natural and expressive. onnxruntime, int8 by default |
| **SAPI** | 0 | fast | The voices already in Windows, via PowerShell. Robotic but zero-download |
| **silent** | 0 | – | Speech output off; speech input still works |

Piper and Kokoro both need an espeak-ng phonemiser, and both wheels bundle one
(`piper/espeak-ng-data`, `espeakng_loader/espeak-ng.dll` + data). `_find_espeak_data()`
locates whichever is present, so there is nothing to compile or install
system-wide — which is the main reason these two engines were chosen over
Coqui/XTTS or anything needing a PyTorch install.

Before text reaches a voice it goes through `strip_for_speech()`: headings,
bullets, tables, code fences, links, emphasis and HTML are converted to something
speakable, then `expand_symbols()` rewrites `%`, `&`, `→`, `€` and friends, which
espeak-ng otherwise pronounces badly or not at all.

---

## Voice activity detection

No silero-vad, no webrtcvad — both mean extra native wheels on Windows, and for
"has this person stopped talking?" an energy gate is enough. `SpeechGate`:

1. **Calibrates** for `calibration_frames` chunks (~0.45 s), taking the 40th
   percentile of measured levels as the noise floor. Without this there is
   nothing to compare against: a noisy room and a quiet voice look identical to a
   fixed threshold.
2. **Detects** speech at `floor + adaptive_margin_db` (9 dB default), keeping
   `pre_speech_pad_ms` of pre-roll so the first phoneme is not clipped.
3. **Cuts** after `silence_ms` (900 ms) of trailing quiet, trimming only the
   trailing silence — pauses inside an utterance are natural speech and survive.
4. **Tracks** the floor continuously on quiet frames, so a fan switching on
   mid-conversation does not lock the detector at a stale value.
5. Ignores utterances shorter than `min_speech_ms`, and hard-stops at
   `max_record_s`.

Push-to-talk remains the default because it is unambiguous. Hands-free is opt-in.

---

## Configuration

`config.py` is dataclasses, one nested section per concern, persisted as a single
JSON document. Three properties are worth calling out:

* **Forward/backward compatible loading.** `from_dict()` merges the file over the
  defaults, so adding an option never breaks an existing config, and an unknown
  key in the file is ignored. A file that fails to parse is moved to
  `config.corrupt.json` and rebuilt, rather than making the app unstartable.
* **Write-side validation.** `save()` runs `_jsonable()` against the *declared*
  defaults. A widget handing back a non-JSON object degrades that one field to its
  default instead of writing a file that will not load next time, or crashing on
  exit. (The declared default, not the current value — the current value is
  exactly what may be corrupt.)
* **Tiers.** `low` / `balanced` / `high` are dictionaries of dotted keys applied
  by `apply_tier()`, localised for the configured language
  (`base.en` → `base`, `en_US-lessac-high` → `de_DE-thorsten-high`). A test walks
  every key in every tier and calls `cfg.set()` on it, so a typo in a preset
  fails the build instead of silently doing nothing.

---

## Where files live

`paths.py`, all overridable with `OLLAVOICE_HOME` for a portable install:

| Windows | Contents |
|---|---|
| `%APPDATA%\OllaVoice\config.json` | settings |
| `%APPDATA%\OllaVoice\logs\` | `gui-YYYYMMDD.log`, `crash.log` |
| `%APPDATA%\OllaVoice\transcripts\` | opt-in JSONL chat history, one file per day |
| `%LOCALAPPDATA%\OllaVoice\models\whisper\` | faster-whisper HF cache |
| `%LOCALAPPDATA%\OllaVoice\models\piper\` | `<voice>.onnx` + `.onnx.json` |
| `%LOCALAPPDATA%\OllaVoice\models\kokoro\` | checkpoint + `voices-v1.0.bin` |
| `%LOCALAPPDATA%\OllaVoice\audio\` | scratch WAVs, deleted after use |

Config is roaming-style (small, worth syncing), models are local-style (large,
re-downloadable). `ollavoice bundle` copies the model tree into a portable folder
for an air-gapped machine.

---

## Testing strategy

250+ tests, three layers:

1. **Pure logic** — config/tiers, sentence splitting, VAD state machine, WAV
   maths, hardware tiering, download resume/checksum/mirror logic. No models, no
   audio, no network.
2. **Fake Ollama** — `tests/conftest.py` runs a `ThreadingHTTPServer` that speaks
   enough of the Ollama API (streaming NDJSON chat, tags, ps, pull) to drive the
   pipeline for real. Combined with stub STT/TTS engines this tests the whole
   orchestration deterministically: event ordering, sentence-level streaming,
   history rollback on failure, cancellation, the HTTP API, multipart parsing.
3. **Real models** — `tests/test_integration.py` synthesises with actual Piper and
   transcribes the result with actual Whisper, then asserts that at least 60% of
   the words survive the round trip. Skipped unless the models are present or
   `OLLAVOICE_IT=1`.

The GUI is tested against `tests/qt_stub/`, a permissive fake `PySide6`. It
cannot validate Qt semantics, but constructing the window and firing every event
handler through it catches the mistakes that actually ship: typos in method or
signal names, missing imports, wrong arity, and anything that raises during
construction. With real Qt installed the same tests build the actual window on the
`offscreen` platform. That is how a Qt app gets tested in a headless container.
