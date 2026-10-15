"""``ollavoice chat`` - a voice conversation in your terminal.

Same pipeline as the desktop app, no Qt required: the microphone is read through
whatever audio backend is available (sounddevice, Qt, or ffmpeg/arecord) and the
answer is printed as it streams while the local voice speaks it sentence by
sentence.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from . import audioio, paths as _paths
from .cli import bold, cyan, dim, green, red, yellow
from .config import Config
from .pipeline import PipelineError, VoicePipeline
from .wav import AudioClip

HELP = f"""
{bold('Terminal voice chat')} - everything below runs on this machine.

  {cyan('Enter')}            hold to record from the microphone, press Enter again to send
  {cyan('/text <msg>')}      send a typed message
  {cyan('/file <path>')}    transcribe a WAV file and ask about it
  {cyan('/speak <text>')}   synthesise speech without asking the model
  {cyan('/voice')}          toggle speech output on/off
  {cyan('/listen')}         toggle hands-free listening (auto end-of-speech)
  {cyan('/model <name>')}   switch the Ollama model
  {cyan('/tts <engine>')}   piper | kokoro | sapi | none
  {cyan('/voice-list')}     list installed voices
  {cyan('/stt <engine>')}   faster-whisper | whisper-cpp | ollama | windows
  {cyan('/lang <code>')}     set the spoken language (en, de, zh, ...)
  {cyan('/rate <0.5-2>')}   speaking rate
  {cyan('/system <text>')}  replace the system prompt
  {cyan('/new')}            clear the conversation
  {cyan('/save <file>')}    save the transcript
  {cyan('/doctor')}         health report
  {cyan('/gui')}            launch the desktop app
  {cyan('/quit')}           exit
"""


class TermUI:
    """Renders pipeline events as terminal output."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._streaming = False
        self._token_count = 0
        self._spin_index = 0
        self._last_spin = 0.0

    # ------------------------------------------------------------------ #
    def on_event(self, event) -> None:
        kind = event.kind
        if kind == "phase":
            self._phase(event.phase)
        elif kind == "transcript":
            self._end_stream()
            print(f"{bold('You')}" + dim(f" [{self._meta(event.data)}]") + f": {event.text}")
        elif kind == "token":
            self._token(event.text)
        elif kind == "sentence":
            pass
        elif kind == "audio":
            dur = event.data.get("duration_s", 0)
            synth = event.data.get("synth_s", 0)
            sys.stdout.write(dim(f"\r  [spoke {dur:.1f}s in {synth:.1f}s]".ljust(46)))
            sys.stdout.flush()
        elif kind == "stats":
            pass
        elif kind == "done":
            self._end_stream(final=True)
        elif kind == "error":
            self._end_stream()
            print(red(f"error: {event.text}"))
        elif kind == "info":
            if not self._streaming:
                sys.stdout.write(dim(f"  {event.text}\n"))
                sys.stdout.flush()
        elif kind == "level":
            self._spinner()

    # ------------------------------------------------------------------ #
    def _phase(self, phase: str) -> None:
        labels = {"listening": "listening...", "transcribing": "transcribing...",
                  "thinking": "thinking...", "speaking": "speaking...", "idle": ""}
        text = labels.get(phase, phase)
        if text and not self._streaming:
            sys.stdout.write(dim(f"\r  {text}".ljust(40)))
            sys.stdout.flush()

    @staticmethod
    def _meta(data: dict[str, Any]) -> str:
        bits = []
        if data.get("audio_s"):
            bits.append(f"{data['audio_s']}s audio")
        if data.get("decode_s"):
            bits.append(f"{data['decode_s']}s")
        if data.get("model"):
            bits.append(str(data["model"]))
        return " ".join(bits)

    def _token(self, delta: str) -> None:
        if not self._streaming:
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.write(bold("Ollama") + ": ")
            self._streaming = True
        sys.stdout.write(delta)
        sys.stdout.flush()
        self._token_count += 1

    def _end_stream(self, final: bool = False) -> None:
        if self._streaming:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif final:
            sys.stdout.write("\r" + " " * 60 + "\r")
        self._streaming = False
        self._token_count = 0

    def _spinner(self) -> None:
        now = time.time()
        if now - self._last_spin < 0.12:
            return
        self._last_spin = now
        frames = "|/-\\"
        self._spin_index = (self._spin_index + 1) % len(frames)
        sys.stdout.write(dim(f"\r  {frames[self._spin_index]}"))
        sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Recording in a terminal
# --------------------------------------------------------------------------- #


def record_push_to_talk(cfg: Config, backend: str = "auto", *,
                        hands_free: bool = False) -> AudioClip:
    """Record until the user presses Enter (or until silence in hands-free mode)."""
    if hands_free:
        return _record_hands_free(cfg, backend)

    rec = audioio.make_recorder(backend, sample_rate=cfg.audio.sample_rate,
                                channels=cfg.audio.channels,
                                chunk_ms=cfg.audio.chunk_ms,
                                device=cfg.audio.input_device)
    peak = {"db": -120.0}

    def on_chunk(chunk) -> None:
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0))))
        db = 20.0 * float(np.log10(rms)) if rms > 1e-7 else -120.0
        peak["db"] = max(peak["db"], db * 0.2 + peak["db"] * 0.8)

    rec.on_chunk = on_chunk
    rec.start()
    print(dim("  recording... press Enter to send, Ctrl+C to discard"), flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        rec.abort()
        print()
        import numpy as np

        return AudioClip(np.zeros(0, dtype=np.int16), int(cfg.audio.sample_rate))
    clip = rec.stop()
    if peak["db"] < -55:
        print(yellow("  (that recording was almost silent - is the right microphone selected?)"))
    return clip


def _record_hands_free(cfg: Config, backend: str) -> AudioClip:

    from .vad import SpeechGate, VadSettingsLite

    gate = SpeechGate(VadSettingsLite(
        silence_ms=cfg.vad.silence_ms, min_speech_ms=cfg.vad.min_speech_ms,
        max_record_s=cfg.vad.max_record_s, threshold_db=cfg.vad.threshold_db,
        adaptive=cfg.vad.adaptive, adaptive_margin_db=cfg.vad.adaptive_margin_db,
        pre_speech_pad_ms=cfg.vad.pre_speech_pad_ms,
        calibration_frames=cfg.vad.calibration_frames))
    done = {"stop": False}

    def on_chunk(chunk) -> None:
        for ev in gate.push(chunk, rec.sample_rate):
            if ev.kind == "speech_start":
                sys.stdout.write(dim("\r  speech detected...   "))
                sys.stdout.flush()
            elif ev.kind in ("speech_end", "timeout"):
                done["stop"] = True

    rec = audioio.make_recorder(backend, sample_rate=cfg.audio.sample_rate,
                                channels=cfg.audio.channels,
                                chunk_ms=cfg.audio.chunk_ms,
                                device=cfg.audio.input_device, on_chunk=on_chunk)
    rec.start()
    print(dim("  listening (hands-free)... speak, then pause. Ctrl+C to abort"), flush=True)
    try:
        while not done["stop"] and rec.elapsed_s < cfg.vad.max_record_s + 2:
            time.sleep(0.05)
    except KeyboardInterrupt:
        rec.abort()
        raise
    clip = gate.clip if len(gate.clip.samples) else rec.stop()
    rec.stop()
    return clip


# --------------------------------------------------------------------------- #


def run_repl(cfg: Config, *, voice_in: bool = True, hands_free: bool = False,
             audio_backend: str = "auto", one_shot: str = "") -> int:
    ui = TermUI(cfg)
    try:
        backend = audioio.choose_backend(audio_backend)
    except audioio.AudioError as exc:
        backend = "system"
        print(yellow(f"{exc}\nContinuing without a microphone: you can still type."))
        voice_in = False

    pipeline = VoicePipeline(cfg, audio_backend=backend, event_fn=ui.on_event)

    print(bold("OllaVoice terminal chat"))
    print(dim(f"  ollama : {cfg.ollama.host} model={cfg.ollama.model or '(not set)'}"))
    print(dim(f"  ear    : {cfg.stt.engine}/{cfg.stt.model}"))
    print(dim(f"  mouth  : {cfg.tts.engine}"
              + (f"/{cfg.tts.piper_voice}" if cfg.tts.engine == "piper"
                 else f"/{cfg.tts.kokoro_voice}" if cfg.tts.engine == "kokoro" else "")))
    print(dim(f"  audio  : backend={backend}"
              + ("" if voice_in else " (microphone disabled)")))
    print(dim("  type /help for commands, /quit to exit"))

    # Fail early with a useful message rather than after the user has spoken.
    problems = _preflight(cfg, pipeline, voice_in=voice_in)
    for problem in problems:
        print(yellow(f"  ! {problem}"))

    if one_shot:
        try:
            pipeline.run_turn(text=one_shot, speak=cfg.tts.auto_speak)
            pipeline.wait_for_playback()
            return 0
        except PipelineError as exc:
            print(red(str(exc)))
            return 1

    if cfg.tts.engine not in ("none", ""):
        try:
            pipeline.warm()
        except Exception:
            pass

    while True:
        try:
            prompt = green("mic>") if voice_in else green(">")
            line = input(f"\n{prompt} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            if not voice_in:
                continue
            if pipeline.is_busy:
                print(dim("  still working..."))
                continue
            try:
                clip = record_push_to_talk(cfg, backend, hands_free=hands_free)
            except audioio.AudioError as exc:
                print(red(f"  microphone error: {exc}"))
                continue
            except KeyboardInterrupt:
                print()
                continue
            if clip.duration_s < 0.3:
                print(yellow("  nothing recorded"))
                continue
            try:
                pipeline.run_turn(clip=clip, speak=cfg.tts.auto_speak,
                                  blocking_playback=False)
                pipeline.wait_for_playback()
                ui._end_stream(final=True)
            except PipelineError as exc:
                print(red(f"  {exc}"))
            except KeyboardInterrupt:
                pipeline.cancel()
                print(dim("\n  interrupted"))
            continue

        if line.startswith("/"):
            stop = _handle_command(line, cfg, pipeline, ui, backend)
            if stop:
                break
            continue

        try:
            pipeline.run_turn(text=line, speak=cfg.tts.auto_speak, blocking_playback=False)
            pipeline.wait_for_playback()
            ui._end_stream(final=True)
        except PipelineError as exc:
            print(red(f"  {exc}"))
        except KeyboardInterrupt:
            pipeline.cancel()
            print(dim("\n  interrupted"))

    print(dim("bye"))
    return 0


def _preflight(cfg: Config, pipeline: VoicePipeline, *, voice_in: bool) -> list[str]:
    problems: list[str] = []
    from .ollama_client import OllamaClient

    client = OllamaClient(cfg.ollama.host, cfg.ollama.api_key, timeout=4)
    if not client.is_alive():
        problems.append(f"Ollama is not reachable at {cfg.ollama.host} - start it with "
                        "`ollama serve` or from the Start menu")
    elif not cfg.ollama.model:
        problems.append("no model configured - use /model <name> or `ollavoice setup`")
    elif not client.has_model(cfg.ollama.model):
        problems.append(f"model '{cfg.ollama.model}' is not pulled - run: "
                        f"ollama pull {cfg.ollama.model}")
    if voice_in:
        from .stt_engines import available_engines

        engines = {e["name"]: e for e in available_engines()}
        info = engines.get(cfg.stt.engine, {})
        if not info.get("available"):
            problems.append(f"STT engine '{cfg.stt.engine}' is not available: {info.get('hint','')}")
    from .tts_engines import available_engines as tts_available

    engines = {e["name"]: e for e in tts_available()}
    info = engines.get(cfg.tts.engine, {})
    if not info.get("available"):
        problems.append(f"TTS engine '{cfg.tts.engine}' is not available: {info.get('hint','')}")
    return problems


def _handle_command(line: str, cfg: Config, pipeline: VoicePipeline, ui: TermUI,
                    backend: str) -> bool:
    """Returns True when the REPL should exit."""
    from .config import save as save_config

    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("q", "quit", "exit"):
        return True
    if cmd in ("h", "help", "?"):
        print(HELP)
    elif cmd in ("text", "ask", "t"):
        if not arg:
            print(yellow("  usage: /text your message"))
        else:
            try:
                pipeline.run_turn(text=arg, speak=cfg.tts.auto_speak, blocking_playback=False)
                pipeline.wait_for_playback()
                ui._end_stream(final=True)
            except PipelineError as exc:
                print(red(f"  {exc}"))
    elif cmd == "file":
        if not arg or not os.path.exists(arg):
            print(yellow("  usage: /file path/to/recording.wav"))
        else:
            try:
                from .wav import read_wav

                clip = read_wav(arg)
                pipeline.run_turn(clip=clip, speak=cfg.tts.auto_speak, blocking_playback=False)
                pipeline.wait_for_playback()
                ui._end_stream(final=True)
            except Exception as exc:
                print(red(f"  {exc}"))
    elif cmd == "speak":
        if not arg:
            print(yellow("  usage: /speak some text"))
        else:
            try:
                pipeline.speak_text(arg, blocking=False)
                pipeline.wait_for_playback()
            except Exception as exc:
                print(red(f"  {exc}"))
    elif cmd == "voice":
        cfg.tts.auto_speak = not cfg.tts.auto_speak
        save_config(cfg)
        print(green(f"  speech output: {'on' if cfg.tts.auto_speak else 'off'}"))
    elif cmd == "listen":
        cfg.vad.enabled = not cfg.vad.enabled
        save_config(cfg)
        print(green(f"  hands-free listening: {'on' if cfg.vad.enabled else 'off'}"))
    elif cmd == "model":
        if not arg:
            names = pipeline.ollama.model_names()
            print(dim("  installed: ") + (", ".join(names) or "(none)"))
            print(dim(f"  current  : {cfg.ollama.model}"))
        else:
            cfg.ollama.model = arg
            pipeline.config = cfg
            save_config(cfg)
            print(green(f"  model: {arg}"))
    elif cmd == "tts":
        if arg:
            cfg.tts.engine = arg
            pipeline.reload_engines()
            save_config(cfg)
            print(green(f"  tts engine: {arg}"))
            try:
                pipeline.warm(stt=False)
            except Exception as exc:
                print(red(f"  {exc}"))
        else:
            from .tts_engines import available_engines

            for e in available_engines():
                mark = green("*") if e["name"] == cfg.tts.engine else " "
                print(f"  {mark} {e['name']:<10}{dim(e['description'])}")
    elif cmd in ("voices", "voice-list"):
        from .tts_engines import installed_voices

        info = installed_voices()
        print(dim("  piper : ") + (", ".join(info["piper"]) or "(none installed)"))
        print(dim("  kokoro: ") + (", ".join(info["kokoro"]) or "(none installed)"))
        print(dim("  get more with: ollavoice voices --install en_GB-alan-medium"))
    elif cmd == "stt":
        if arg:
            cfg.stt.engine = arg
            pipeline.reload_engines()
            save_config(cfg)
            print(green(f"  stt engine: {arg}"))
        else:
            from .stt_engines import available_engines

            for e in available_engines():
                mark = green("*") if e["name"] == cfg.stt.engine else " "
                print(f"  {mark} {e['name']:<16}{dim(e['description'])}")
    elif cmd == "lang":
        if not arg:
            print(dim(f"  language: {cfg.language} (stt: {cfg.stt.language})"))
        else:
            from .config import localize_piper_voice, resolve_whisper_model

            cfg.language = arg
            cfg.stt.language = arg
            cfg.stt.model = resolve_whisper_model(cfg.stt.model, arg)
            if cfg.tts.engine == "piper":
                cfg.tts.piper_voice = localize_piper_voice(cfg.tts.piper_voice, arg)
            pipeline.reload_engines()
            save_config(cfg)
            print(green(f"  language: {arg} (whisper {cfg.stt.model}, "
                        f"voice {cfg.tts.piper_voice})"))
    elif cmd == "rate":
        try:
            cfg.tts.rate = max(0.5, min(2.0, float(arg)))
            pipeline.reload_engines()
            save_config(cfg)
            print(green(f"  speaking rate: {cfg.tts.rate:.2f}x"))
        except ValueError:
            print(yellow("  usage: /rate 1.15"))
    elif cmd == "system":
        cfg.ollama.system_prompt = arg
        pipeline.history.system = arg
        save_config(cfg)
        print(green("  system prompt updated"))
    elif cmd == "new":
        pipeline.history.clear()
        print(green("  conversation cleared"))
    elif cmd == "save":
        target = arg or str(_paths.transcripts_dir() / f"chat-{time.strftime('%Y%m%d-%H%M')}.txt")
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(pipeline.history.as_transcript() + "\n")
            print(green(f"  saved {target}"))
        except OSError as exc:
            print(red(f"  {exc}"))
    elif cmd == "status":
        import json as _json

        print(dim(_json.dumps(pipeline.status(), indent=2, default=str)))
    elif cmd == "doctor":
        from . import status_report

        print(status_report())
    elif cmd == "gui":
        print(dim("  launching the desktop app..."))
        os.execv(sys.executable, [sys.executable, "-m", "ollavoice", "gui"])
    elif cmd == "devices":
        for d in audioio.list_devices(backend):
            mark = "*" if d.name == cfg.audio.input_device else " "
            print(f"  {mark} [{d.kind}] {d.label()}")
    elif cmd in ("mic", "input"):
        cfg.audio.input_device = arg
        save_config(cfg)
        print(green(f"  microphone: {arg or '(system default)'}"))
    else:
        print(yellow(f"  unknown command '/{cmd}' - try /help"))
    return False
