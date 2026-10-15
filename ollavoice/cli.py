"""``ollavoice`` - the command line front-end.

    ollavoice chat              voice conversation in your terminal
    ollavoice gui               launch the native desktop app
    ollavoice setup             interactive first-run setup (or --yes to auto)
    ollavoice doctor            health report: hardware, Ollama, engines, models
    ollavoice listen            record -> print the transcript
    ollavoice transcribe FILE   transcribe an existing WAV/MP3/FLAC
    ollavoice speak "text"      synthesise speech (and optionally write a WAV)
    ollavoice voices            list installed + downloadable voices
    ollavoice serve             local speech API (OpenAI-compatible endpoints)
    ollavoice config get|set    inspect/change settings

Run ``ollavoice <command> --help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, paths as _paths
from .config import TIER_ORDER, Config, apply_tier, load as load_config, save as save_config


# --------------------------------------------------------------------------- #
# Small console helpers
# --------------------------------------------------------------------------- #

def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(t: str) -> str: return _c("2", t)
def bold(t: str) -> str: return _c("1", t)
def green(t: str) -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str) -> str: return _c("31", t)
def cyan(t: str) -> str: return _c("36", t)


def _progress_bar(done: float, total: float, width: int = 34) -> str:
    if total <= 0:
        return f"{done / 1e6:7.1f} MB"
    frac = min(1.0, done / total)
    filled = int(width * frac)
    return f"[{'#' * filled}{'-' * (width - filled)}] {frac * 100:5.1f}% {done / 1e6:6.1f}/{total / 1e6:.1f} MB"


def _make_progress(label: str):
    state = {"last": 0.0}

    def progress(done: float, total: float, name: str) -> None:
        now = time.time()
        if now - state["last"] < 0.15 and total and done < total:
            return
        state["last"] = now
        sys.stdout.write("\r" + f"  {label} {name[:28]:<28} " + _progress_bar(done, total))
        sys.stdout.flush()
        if total and done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return progress


# --------------------------------------------------------------------------- #
# Root parser
# --------------------------------------------------------------------------- #

#: Printed at the end of ``--help``. argparse already lists the subcommands under
#: "positional arguments", so this only adds what that list cannot: the two
#: commands people look for first, and the privacy guarantee.
COMMAND_EPILOG = """
Start here:
  ollavoice setup            detect hardware, fetch speech models, verify the loop
  ollavoice gui              launch the desktop app (hold Space to talk)
  ollavoice chat             the same thing in your terminal
  ollavoice doctor --test    synthesise a sentence and transcribe it back

Everything is local: no cloud API, no telemetry, no account.
Run `ollavoice <command> --help` for the full flag list.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollavoice",
        description="Local speech input and speech output for Ollama. "
                    "Everything runs on your own machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=COMMAND_EPILOG,
    )
    p.add_argument("--version", action="version", version=f"OllaVoice {__version__}")
    p.add_argument("--config", type=Path, default=None, help="use an alternate config file")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    # chat -----------------------------------------------------------------
    c = sub.add_parser("chat", help="voice + text conversation in the terminal")
    c.add_argument("-m", "--model", help="Ollama model to use")
    c.add_argument("--no-voice-in", action="store_true", help="disable the microphone")
    c.add_argument("--no-voice-out", action="store_true", help="disable speech output")
    c.add_argument("--hands-free", action="store_true",
                   help="listen continuously and auto-detect end of sentence")
    c.add_argument("--audio-backend", default="auto",
                   choices=["auto", "qt", "sounddevice", "system", "null"])
    c.add_argument("-p", "--prompt", help="send one prompt, print/speak the answer, exit")
    c.set_defaults(func=cmd_chat)

    # gui ------------------------------------------------------------------
    g = sub.add_parser("gui", help="launch the native desktop app")
    g.add_argument("--minimized", action="store_true", help="start in the tray")
    g.add_argument("--reset-window", action="store_true", help="forget the saved window size")
    g.set_defaults(func=cmd_gui)

    # setup ----------------------------------------------------------------
    s = sub.add_parser("setup", help="download speech models and check Ollama")
    s.add_argument("--tier", choices=[*TIER_ORDER, "custom"], help="low | balanced | high")
    s.add_argument("--language", default=None, help="spoken language, e.g. en, de, zh")
    s.add_argument("--model", help="Ollama chat model to use/pull")
    s.add_argument("--stt-engine", default=None,
                   choices=["faster-whisper", "whisper-cpp", "ollama", "windows"])
    s.add_argument("--stt-model", default=None, help="Whisper size, e.g. base.en / small.en")
    s.add_argument("--tts-engine", default=None, choices=["piper", "kokoro", "sapi", "none"])
    s.add_argument("--tts-voice", default=None, help="Piper voice, e.g. en_US-lessac-high")
    s.add_argument("--kokoro-voice", default=None, help="Kokoro voice, e.g. af_heart")
    s.add_argument("--pull", action="store_true", help="also pull the Ollama chat model")
    s.add_argument("--no-pull", action="store_true", help="never pull, only configure")
    s.add_argument("--yes", "-y", action="store_true", help="non-interactive")
    s.add_argument("--offline", action="store_true",
                   help="install from the local vendor bundle only; never touch the network")
    s.add_argument("--skip-stt", action="store_true")
    s.add_argument("--skip-tts", action="store_true")
    s.set_defaults(func=cmd_setup)

    # doctor ---------------------------------------------------------------
    d = sub.add_parser("doctor", help="health report + engine self-tests")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.add_argument("--test", action="store_true",
                   help="actually synthesise a sentence and transcribe it back")
    d.set_defaults(func=cmd_doctor)

    # listen ---------------------------------------------------------------
    l = sub.add_parser("listen", help="record from the microphone and print the transcript")
    l.add_argument("-d", "--duration", type=float, default=0,
                   help="fixed recording length in seconds (default: press Enter to stop)")
    l.add_argument("--file", type=Path, help="transcribe this audio file instead of the mic")
    l.add_argument("-o", "--output", type=Path, help="also write the transcript to a file")
    l.add_argument("--save-audio", type=Path, help="keep the recording as WAV")
    l.add_argument("--audio-backend", default="auto",
                   choices=["auto", "qt", "sounddevice", "system", "null"])
    l.add_argument("--ask", action="store_true",
                   help="send the transcript to Ollama and print/speak the answer")
    l.set_defaults(func=cmd_listen)

    # transcribe -----------------------------------------------------------
    t = sub.add_parser("transcribe", help="transcribe an existing audio file")
    t.add_argument("file", type=Path)
    t.add_argument("-o", "--output", type=Path, help="write text here (default: stdout)")
    t.add_argument("--timestamps", action="store_true", help="include per-segment timings")
    t.add_argument("--language", default=None)
    t.set_defaults(func=cmd_transcribe)

    # speak ----------------------------------------------------------------
    sp = sub.add_parser("speak", help="read text aloud with the local TTS voice")
    sp.add_argument("text", nargs="*", help="text to speak ('-' or empty reads stdin)")
    sp.add_argument("-f", "--file", type=Path, help="read text from a file")
    sp.add_argument("-o", "--output", type=Path, help="write a WAV instead of playing")
    sp.add_argument("--engine", choices=["piper", "kokoro", "sapi", "none"])
    sp.add_argument("--voice", help="Piper voice name, e.g. en_GB-alan-medium")
    sp.add_argument("--kokoro-voice", help="Kokoro voice name, e.g. am_michael")
    sp.add_argument("--rate", type=float, help="speaking rate (0.5 - 2.0)")
    sp.add_argument("--audio-backend", default="auto",
                    choices=["auto", "qt", "sounddevice", "system", "null"])
    sp.set_defaults(func=cmd_speak)

    # voices ---------------------------------------------------------------
    v = sub.add_parser("voices", help="list installed and downloadable TTS voices")
    v.add_argument("--all", action="store_true", help="also list the whole Piper catalogue")
    v.add_argument("--install", metavar="VOICE", help="download a Piper voice by name")
    v.add_argument("--language", help="filter the catalogue, e.g. en, de, zh")
    v.set_defaults(func=cmd_voices)

    # models ---------------------------------------------------------------
    m = sub.add_parser("models", help="list local Ollama models")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_models)

    # serve ----------------------------------------------------------------
    sv = sub.add_parser("serve", help="local speech API + browser UI")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--no-browser", action="store_true")
    sv.set_defaults(func=cmd_serve)

    # config ---------------------------------------------------------------
    cf = sub.add_parser("config", help="read/write the config file")
    cf_sub = cf.add_subparsers(dest="cfg_cmd")
    cf_get = cf_sub.add_parser("get", help="print one value, or the whole config")
    cf_get.add_argument("key", nargs="?", help="dotted key, e.g. tts.piper_voice")
    cf_set = cf_sub.add_parser("set", help="set one value")
    cf_set.add_argument("key")
    cf_set.add_argument("value")
    cf_sub.add_parser("show", help="print the whole config")
    cf_sub.add_parser("path", help="print the config file location")
    cf_sub.add_parser("reset", help="delete the config and rebuild defaults")
    cf.add_argument("--tier", choices=TIER_ORDER, help="apply a preset tier")
    cf.set_defaults(func=cmd_config)

    # bundle ---------------------------------------------------------------
    b = sub.add_parser("bundle", help="create an offline model bundle for another PC")
    b.add_argument("dest", type=Path, help="folder to write the bundle into")
    b.add_argument("--wheels", type=Path, help="include wheels from this directory")
    b.set_defaults(func=cmd_bundle)

    return p


def _cfg(args: argparse.Namespace) -> Config:
    cfg = load_config(getattr(args, "config", None), create_if_missing=True)
    if getattr(args, "model", None):
        cfg.ollama.model = args.model
    return cfg


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_doctor(args: argparse.Namespace) -> int:
    from . import status_report
    from .config import load as _load

    cfg = _load(getattr(args, "config", None), create_if_missing=False)
    if args.json:
        print(json.dumps(status_report(as_dict=True), indent=2, default=str))
        return 0
    print(status_report())
    if not args.test:
        print(dim("\nTip: `ollavoice doctor --test` synthesises a sentence and transcribes "
                  "it back to prove the whole loop works."))
        return 0

    print(bold("\n[self-test]"))
    ok = True
    # TTS round trip
    try:
        from .tts_engines import create_engine as create_tts

        t0 = time.time()
        tts = create_tts(cfg.tts, language=cfg.language, auto_download=False)
        tts.load()
        res = tts.synthesize("Self test. The local voice engine is working.")
        print(f"  {green('ok')}   TTS {cfg.tts.engine}/{tts.voice_name}: "
              f"{res.clip.duration_s:.2f}s of audio in {time.time() - t0:.2f}s "
              f"({res.realtime_factor:.2f}x realtime)")
        test_wav = _paths.audio_tmp_dir() / "selftest.wav"
        res.clip.write_wav(test_wav)
    except Exception as exc:
        ok = False
        test_wav = None
        print(f"  {red('fail')} TTS: {exc}")

    # STT round trip
    if test_wav is not None:
        try:
            from .stt_engines import create_engine as create_stt

            stt = create_stt(cfg.stt, ollama_host=cfg.ollama.host)
            tr = stt.transcribe(test_wav)
            verdict = green("ok") if tr.text.strip() else yellow("warn")
            print(f"  {verdict}   STT {tr.engine}/{tr.model}: {tr.text.strip()!r} "
                  f"in {tr.decode_s:.2f}s ({tr.realtime_factor:.2f}x realtime)")
            test_wav.unlink(missing_ok=True)
        except Exception as exc:
            ok = False
            print(f"  {red('fail')} STT: {exc}")

    # Ollama round trip
    try:
        from .ollama_client import OllamaClient

        client = OllamaClient(cfg.ollama.host, cfg.ollama.api_key, timeout=60)
        if client.is_alive():
            if cfg.ollama.model and client.has_model(cfg.ollama.model):
                t0 = time.time()
                answer = client.chat_once(
                    [{"role": "user", "content": "Reply with the single word: ready"}],
                    cfg.ollama.model, options={"num_predict": 8, "temperature": 0})
                print(f"  {green('ok')}   Ollama {cfg.ollama.model}: {answer.strip()!r} "
                      f"in {time.time() - t0:.1f}s")
            else:
                ok = False
                print(f"  {red('fail')} Ollama is online but '{cfg.ollama.model}' is not "
                      f"pulled. Run: ollama pull {cfg.ollama.model}")
        else:
            ok = False
            print(f"  {red('fail')} Ollama is not reachable at {cfg.ollama.host}")
    except Exception as exc:
        ok = False
        print(f"  {red('fail')} Ollama: {exc}")

    return 0 if ok else 1


def cmd_setup(args: argparse.Namespace) -> int:
    from .setup import run_setup

    return run_setup(args)


def cmd_models(args: argparse.Namespace) -> int:
    from .ollama_client import OllamaClient, OllamaError

    cfg = _cfg(args)
    client = OllamaClient(cfg.ollama.host, cfg.ollama.api_key, timeout=10)
    try:
        models = client.list_models()
    except OllamaError as exc:
        print(red(str(exc)))
        return 2
    if args.json:
        print(json.dumps(models, indent=2))
        return 0
    if not models:
        print(yellow("No models yet. Try: ollama pull llama3.2:3b"))
        return 0
    print(f"{'NAME':<34}{'SIZE':>10}{'MODIFIED':>16}")
    for m in models:
        size = (m.get("size") or 0) / 1e9
        modified = time.strftime("%Y-%m-%d", time.localtime(m.get("modified_at") or 0)) \
            if m.get("modified_at") else ""
        print(f"{m.get('name',''):<34}{size:>9.1f}G{modified:>16}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .stt_engines import SttError, create_engine
    from .wav import read_wav

    cfg = _cfg(args)
    if args.language:
        cfg.stt.language = args.language
    if not args.file.exists():
        print(red(f"No such file: {args.file}"))
        return 2
    try:
        engine = create_engine(cfg.stt, ollama_host=cfg.ollama.host,
                               app_language=cfg.language)
    except SttError as exc:
        print(red(str(exc)))
        return 2

    print(dim(f"transcribing {args.file.name} with {cfg.stt.engine}/{cfg.stt.model} ..."),
          file=sys.stderr)
    t0 = time.time()
    try:
        result = engine.transcribe(read_wav(args.file) if args.file.suffix.lower() == ".wav"
                                   else args.file)
    except Exception as exc:
        print(red(f"transcription failed: {exc}"))
        return 1
    text = result.text.strip()
    if args.timestamps:
        lines = [f"[{s.start:7.2f} -> {s.end:7.2f}] {s.text}" for s in result.segments]
        text = "\n".join(lines) or text
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(dim(f"wrote {args.output}"), file=sys.stderr)
    else:
        print(text)
    print(dim(f"{result.summary()} | wall {time.time() - t0:.2f}s"), file=sys.stderr)
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    from . import audioio
    from .pipeline import VoicePipeline
    from .vad import split_on_silence

    cfg = _cfg(args)
    events: list[Any] = []

    def on_event(ev) -> None:
        events.append(ev)
        if ev.kind == "level":
            return
        if ev.kind == "error":
            print(red(ev.text), file=sys.stderr)
        elif ev.kind == "info":
            print(dim(ev.text), file=sys.stderr)

    if args.file:
        from .wav import read_wav

        try:
            clip = read_wav(args.file)
        except Exception:
            clip = None
        if clip is None:
            print(red("Only 16-bit WAV input is supported for --file. Convert first, e.g.\n"
                      "  ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav"))
            return 2
    else:
        print(bold("Recording... "), end="", flush=True)
        if args.duration > 0:
            print(dim(f"({args.duration:.0f}s)"), flush=True)
            clip = audioio.record_clip(args.duration, backend=args.audio_backend,
                                       sample_rate=cfg.audio.sample_rate,
                                       device=cfg.audio.input_device)
        else:
            print(dim("(press Enter to stop)"), flush=True)
            rec = audioio.make_recorder(args.audio_backend,
                                        sample_rate=cfg.audio.sample_rate,
                                        device=cfg.audio.input_device,
                                        chunk_ms=cfg.audio.chunk_ms)
            rec.start()
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            clip = rec.stop()
        print(dim(f"captured {clip.duration_s:.1f}s"), file=sys.stderr)

    if args.save_audio:
        clip.write_wav(args.save_audio)
        print(dim(f"saved {args.save_audio}"), file=sys.stderr)

    if clip.duration_s < 0.25:
        print(yellow("Recording was too short - nothing to transcribe."))
        return 1

    pipe = VoicePipeline(cfg, audio_backend=args.audio_backend, event_fn=on_event)
    parts = split_on_silence(clip) if clip.duration_s > 25 else [clip]
    texts = []
    for i, part in enumerate(parts, 1):
        if len(parts) > 1:
            print(dim(f"-- utterance {i}/{len(parts)} ({part.duration_s:.1f}s) --"),
                  file=sys.stderr)
        result = pipe.transcribe(part, emit=False)
        print(dim(result.summary()), file=sys.stderr)
        texts.append(result.text.strip())
    text = " ".join(t for t in texts if t).strip()
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(dim(f"wrote {args.output}"), file=sys.stderr)
    print(text)

    if args.ask and text:
        answer = pipe.run_turn(text=text, speak=False)
        print()
        print(bold("Ollama: ") + answer)
    return 0


def cmd_speak(args: argparse.Namespace) -> int:
    from .tts_engines import TtsError, create_engine
    from .wav import AudioClip

    cfg = _cfg(args)
    if args.engine:
        cfg.tts.engine = args.engine
    if args.voice:
        cfg.tts.piper_voice = args.voice
        cfg.tts.engine = cfg.tts.engine if args.engine else "piper"
    if args.kokoro_voice:
        cfg.tts.kokoro_voice = args.kokoro_voice
        cfg.tts.engine = cfg.tts.engine if args.engine else "kokoro"
    if args.rate:
        cfg.tts.rate = args.rate

    text = " ".join(args.text).strip()
    if args.file:
        text = (text + "\n" if text else "") + args.file.read_text(encoding="utf-8")
    if not text or text == "-":
        # Only block on stdin when something is actually piped in, so a bare
        # `ollavoice speak` gives a usage error instead of hanging.
        try:
            if not sys.stdin.isatty():
                text = sys.stdin.read()
        except (OSError, ValueError, AttributeError):
            pass
    text = text.strip()
    if not text:
        print(red("Nothing to speak. Pass text, --file, or pipe it in on stdin."))
        return 2
    if cfg.tts.engine in ("none", ""):
        # The user explicitly asked for silence; that is a valid configuration,
        # not an error, so say so and succeed.
        print(yellow("Speech output is disabled (tts.engine=none)."))
        print(dim("Enable it with: ollavoice config set tts.engine piper"))
        return 0

    try:
        engine = create_engine(cfg.tts, language=cfg.language)
    except TtsError as exc:
        print(red(str(exc)))
        return 2

    print(dim(f"synthesising with {cfg.tts.engine}"
              f"/{getattr(engine, 'voice_name', cfg.tts.piper_voice)} ..."), file=sys.stderr)
    t0 = time.time()
    try:
        engine.load()
        pieces: list[AudioClip] = list(engine.synthesize_stream(text))
    except TtsError as exc:
        print(red(str(exc)))
        return 1
    if not pieces:
        print(red("No audio produced."))
        return 1
    clip = pieces[0]
    for p in pieces[1:]:
        clip = clip.concat(p)
    synth_s = time.time() - t0

    if args.output:
        clip.write_wav(args.output)
        print(dim(f"wrote {args.output} ({clip.duration_s:.1f}s, synth {synth_s:.1f}s)"),
              file=sys.stderr)
        return 0

    from . import audioio

    try:
        audioio.play_clip(clip, backend=args.audio_backend, blocking=True)
    except audioio.AudioError as exc:
        fallback = _paths.audio_tmp_dir() / "speak.wav"
        clip.write_wav(fallback)
        reason = str(exc).strip().rstrip(".")
        print(yellow(f"Could not play audio:\n{reason}\n\nWrote the audio to {fallback} "
                     "instead - open it in any player."))
        return 1
    print(dim(f"spoken {clip.duration_s:.1f}s of audio in {synth_s:.1f}s"), file=sys.stderr)
    return 0


def cmd_voices(args: argparse.Namespace) -> int:
    from . import downloads, tts_engines

    cfg = _cfg(args)
    if args.install:
        target = downloads.ensure_piper_voice(args.install, _paths.piper_dir(),
                                              progress=_make_progress("piper"))
        print(green(f"installed {args.install} -> {target}"))
        return 0

    installed = tts_engines.installed_voices()
    if not args.language:
        print(bold("Installed Piper voices") + dim(f"  ({_paths.piper_dir()})"))
        for v in installed["piper"] or [dim("(none - run: ollavoice setup)")]:
            print(f"  {v}")
        print()
        print(bold("Installed Kokoro models") + dim(f"  ({_paths.kokoro_dir()})"))
        for v in installed["kokoro"] or [dim("(none)")]:
            print(f"  {v}")
        if installed["kokoro"]:
            try:
                from .tts_engines import KokoroTts

                kokoro = KokoroTts(cfg.tts, language=cfg.language, auto_download=False)
                names = kokoro.list_voices()
                print(dim(f"  {len(names)} voices: ") + ", ".join(names))
            except Exception:
                pass
        print()

    if args.all or args.language:
        from .voices import filter_by_language, piper_catalogue

        catalogue = piper_catalogue()
        if not catalogue:
            print(red("Could not read the Piper catalogue - is piper-tts installed?"))
            return 1
        total = len(catalogue)
        if args.language:
            catalogue = filter_by_language(catalogue, args.language)
        print(bold(f"Piper catalogue ({len(catalogue)} of {total} voices"
                   f"{', language=' + args.language if args.language else ''})"))
        for v in catalogue:
            print(f"  {v}")
        example = catalogue[0] if catalogue else "en_GB-alan-medium"
        print(dim(f"\nInstall one with: ollavoice voices --install {example}"))
        print(dim("Browse the whole catalogue at huggingface.co/rhasspy/piper-voices"))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    cfg = _cfg(args)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    cfg.server.enabled = True
    return serve(cfg, host=host, port=port, open_browser=not args.no_browser)


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(getattr(args, "config", None), create_if_missing=True)
    if args.tier:
        apply_tier(cfg, args.tier)
        save_config(cfg, getattr(args, "config", None))
        print(green(f"applied the '{args.tier}' tier"))
        _print_key(cfg, "ollama.model")
        _print_key(cfg, "stt.model")
        _print_key(cfg, "tts.engine")
        _print_key(cfg, "tts.piper_voice")
        return 0

    cmd = args.cfg_cmd or "show"
    if cmd == "path":
        print(_paths.config_file())
        return 0
    if cmd == "reset":
        from .config import default_config

        cfg = default_config()
        save_config(cfg, getattr(args, "config", None))
        print(green("config reset to defaults"))
        return 0
    if cmd == "show":
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    if cmd == "get":
        if not args.key:
            print(json.dumps(cfg.to_dict(), indent=2))
            return 0
        _print_key(cfg, args.key)
        return 0
    if cmd == "set":
        try:
            value = cfg.set(args.key, args.value)
        except (AttributeError, KeyError) as exc:
            print(red(f"unknown key {args.key!r} ({exc})"))
            return 2
        save_config(cfg, getattr(args, "config", None))
        print(green(f"{args.key} = {value!r}"))
        if args.key in ("tier",) or args.key.startswith(("stt.", "tts.", "ollama.model")):
            print(dim("restart the app/CLI for this to take effect"))
        return 0
    return 2


def _print_key(cfg: Config, key: str) -> None:
    try:
        value = cfg.get(key)
    except (AttributeError, KeyError):
        print(red(f"unknown key {key!r}"))
        return
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2))
    else:
        print(value)


def cmd_bundle(args: argparse.Namespace) -> int:
    from .downloads import make_offline_bundle

    dest = make_offline_bundle(args.dest, wheels_dir=args.wheels)
    print(green(f"offline bundle written to {dest}"))
    print(dim("Copy that folder to the offline PC, then unpack its subdirectories into:\n"
              f"  {_paths.data_root()}"))
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:
        print(red(
            "The desktop GUI needs PySide6.\n"
            "  pip install PySide6\n"
            f"(import failed with: {exc})"
        ))
        return 2
    except OSError as exc:
        print(red(
            "PySide6 is installed but its Qt libraries could not be loaded.\n"
            f"  {exc}\n"
            "On Linux: sudo apt install libglib2.0-0 libgl1 libegl1 libxkbcommon0 "
            "libdbus-1-3 libfontconfig1\n"
            "On Windows: pip install --force-reinstall PySide6"
        ))
        return 2
    cfg = _cfg(args)
    if args.reset_window:
        cfg.gui.window_width, cfg.gui.window_height = 980, 700
        save_config(cfg, getattr(args, "config", None))
    return gui_main(cfg, start_minimized=args.minimized)


def cmd_chat(args: argparse.Namespace) -> int:
    from .repl import run_repl

    cfg = _cfg(args)
    if args.no_voice_out:
        cfg.tts.auto_speak = False
    return run_repl(cfg, voice_in=not args.no_voice_in,
                    hands_free=args.hands_free,
                    audio_backend=args.audio_backend,
                    one_shot=args.prompt)


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    _paths.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        print()
        print(bold("Quick start:"))
        print("  1. " + cyan("ollavoice setup") + dim("   (detects your hardware, fetches models)"))
        print("  2. " + cyan("ollavoice doctor --test") + dim("   (proves the loop works)"))
        print("  3. " + cyan("ollavoice gui") + dim("       (or `ollavoice chat` in the terminal)"))
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print(dim("\ninterrupted"))
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
